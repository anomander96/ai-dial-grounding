import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)



SYSTEM_PROMPT = """You are a hobby extraction system. Given user profiles, extract hobbies and group user IDs by hobby.

## Instructions:
- Analyze the `about_me` field of each user
- Extract hobbies and interests mentioned
- Group user IDs by hobby/interest
- Be inclusive - if a hobby is related to the search query, include it
- Return ONLY the JSON, no explanation

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## USER PROFILES:
{context}

## SEARCH QUERY:
{query}"""

class HobbySearchResult(BaseModel):
    hobbies: dict[str, list[int]] = Field(
        default={},
        description="Map of hobby name to list of user IDs that have that hobby"
    )

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small-1",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    dimensions=384
)

llm_client = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    temperature=0.0
)

user_client = UserClient()

vectorstore = Chroma(
    collection_name="users",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def _format_user_document(user: dict[str, Any]) -> str:
    """Format only id and about_me for embedding - keeps vectors small and focused"""
    return f"User ID: {user['id']}\nAbout me: {user.get('about_me', '')}"

async def _initialize_vectorstore(users: list[dict[str, Any]]):
    """Cold start - load all users into vectorstore on first run"""
    print(f"🚀 Cold start: loading {len(users)} users into vectorstore...")

    batch_size = 100
    batches = [users[i:i + batch_size] for i in range(0, len(users), batch_size)]

    for batch in batches:
        documents = [
            Document(
                id=str(user['id']),
                page_content=_format_user_document(user)
            )
            for user in batch
        ]
        await vectorstore.aadd_documents(documents)

    print(f"✅ Vectorstore initialized with {len(users)} users")


async def _update_vectorstore(current_users: list[dict[str, Any]]):
    """Update vectorstore - add new users, remove deleted ones"""
    print("🔄 Updating vectorstore...")

    existing_data = vectorstore.get()
    existing_ids = set(existing_data['ids'])

    current_ids = set(str(user['id']) for user in current_users)

    deleted_ids = existing_ids - current_ids

    new_ids = current_ids - existing_ids

    if deleted_ids:
        print(f"🗑️ Removing {len(deleted_ids)} deleted users")
        vectorstore.delete(ids=list(deleted_ids))

    if new_ids:
        new_users = [u for u in current_users if str(u['id']) in new_ids]
        print(f"➕ Adding {len(new_users)} new users")
        documents = [
            Document(
                id=str(user['id']),
                page_content=_format_user_document(user)
            )
            for user in new_users
        ]
        await vectorstore.aadd_documents(documents)

    print(f"✅ Vectorstore updated (removed: {len(deleted_ids)}, added: {len(new_ids)})")


async def retrieve_context(query: str, k: int = 50) -> str:
    """Search vectorstore for users matching the query"""
    print(f"🔍 Searching for: {query}")

    results = await vectorstore.asimilarity_search(query=query, k=k)

    context = "\n\n".join([doc.page_content for doc in results])
    print(f"📄 Found {len(results)} relevant users")
    return context


def augment_prompt(query: str, context: str) -> str:
    return USER_PROMPT.format(context=context, query=query)


async def generate_answer(augmented_prompt: str) -> dict[str, list[int]]:
    """Ask LLM to extract hobbies and group user IDs"""
    parser = PydanticOutputParser(pydantic_object=HobbySearchResult)

    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt)
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
        format_instructions=parser.get_format_instructions()
    )

    result: HobbySearchResult = (prompt | llm_client | parser).invoke({})
    return result.hobbies


async def output_grounding(hobbies: dict[str, list[int]]) -> dict[str, list[dict]]:
    """
    Fetch full user info for each user ID.
    This also validates IDs are real - like output validation in Java.
    """
    print("🔎 Output grounding - fetching full user details...")

    result = {}

    for hobby, user_ids in hobbies.items():
        users_for_hobby = []
        for user_id in user_ids:
            try:
                user = await user_client.get_user(user_id)
                if user:
                    users_for_hobby.append(user)
            except Exception:
                print(f"⚠️ User ID {user_id} not found, skipping")
                continue

        if users_for_hobby:
            result[hobby] = users_for_hobby

    return result


async def main():
    print("🧙 HOBBIES SEARCHING WIZARD")
    print("Query samples:")
    print("  - I need people who love to go to mountains")
    print("  - Find users interested in music and art")

    # Cold start - load all users into vectorstore
    all_users = user_client.get_all_users()

    # Check if vectorstore is empty - cold start needed
    existing = vectorstore.get()
    if not existing['ids']:
        await _initialize_vectorstore(all_users)
    else:
        # Adaptive update - sync with latest user data
        await _update_vectorstore(all_users)

    while True:
        user_question = input("\n> ").strip()

        if user_question.lower() in ['exit', 'quit']:
            break

        # Before each search - update vectorstore with latest users
        # (users added/deleted every 5 minutes)
        current_users = user_client.get_all_users()
        await _update_vectorstore(current_users)

        # 1. Retrieval
        context = await retrieve_context(query=user_question)

        # 2. Augmentation
        augmented = augment_prompt(query=user_question, context=context)

        # 3. Generation - LLM extracts hobbies and groups user IDs
        hobbies = await generate_answer(augmented)
        print(f"\n🎯 Raw LLM result: {hobbies}")

        if not hobbies:
            print("No users found matching your query.")
            continue

        # 4. Output grounding - fetch full user info and validate IDs
        final_result = await output_grounding(hobbies)

        print("\n📊 Final Result:")
        print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())