import asyncio
import dotenv
import os
from github import Github, Auth, UnknownObjectException
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from typing import Any, Dict, List, Optional

dotenv.load_dotenv()


async def main():
    workflow = build_workflow()
    pr_number = os.getenv("PR_NUMBER")
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)
    handler = workflow.run(prompt.format(), ctx=Context(workflow))

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response and getattr(event.response, "content", None):
                print("\n\nFinal response:", event.response.content)
            if getattr(event, "tool_calls", None):
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


# -------------------- LLM and Agent Setup --------------------
def build_workflow() -> AgentWorkflow:
    llm = init_llm()

    system_prompt = (
        "You are the context gathering agent. When gathering context, you MUST gather \n: "
        "  - The details: author, title, body, diff_url, state, and head_sha; \n"
        "  - Changed files; \n"
        "  - Any requested for files; \n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent. "
    )
    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all the needed context for a PR review (PR details, changed files, any requested repo files) and saves a summary to state.",
        tools=[
            FunctionTool.from_defaults(get_pr_details),
            FunctionTool.from_defaults(get_file_contents),
            FunctionTool.from_defaults(get_commit_details),
            FunctionTool.from_defaults(add_context_to_state)
        ],
        system_prompt=system_prompt,
        can_handoff_to=["CommentorAgent"],
    )

    commentor_system_prompt = (
        "You are the commentor agent responsible for writing a thorough pull request review. "
        "Your primary goal is to analyze the context provided by the ContextAgent and draft a high-quality review comment.\n\n"
        "## Review Content Guidelines:\n"
        "When drafting your review, ensure you cover the following points in markdown format (~200-300 words):\n"
        "1.  **Acknowledge the Author:** Start by addressing the author directly.\n"
        "2.  **Positive Feedback:** Mention what is good about the PR.\n"
        "3.  **Contribution Rules:** Check if the author followed all contribution guidelines. Point out what is missing.\n"
        "4.  **Testing & Migrations:** Use the diff to determine if there are tests for new functionality or migrations for new models.\n"
        "5.  **Documentation:** Use the diff to see if new endpoints are documented.\n"
        "6.  **Specific Suggestions:** Quote specific lines of code that could be improved and offer clear suggestions for implementation.\n\n"
        "## Operational Workflow:\n"
        "- If you do not have enough information to write a complete review, you **MUST** hand off to the `ContextAgent` to request more details.\n"
        "- Once you have all the necessary information and have finished drafting the complete review comment, you **MUST** follow this two-step final procedure:\n"
        "    1. Call the `add_comment_to_state` tool with the full text of your drafted review as the `draft_comment` argument.\n"
        "    2. Immediately after the tool call is complete, you **MUST** hand off control to the `ReviewAndPostingAgent`.\n\n"
        "**CRITICAL:** Do NOT output the review text as your final answer to the user. Your final action is to use the `add_comment_to_state` tool and then hand off."
    )
    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the ContextAgent to create a review comment and save it to the state.",
        tools=[FunctionTool.from_defaults(add_review_comment_to_state)],
        system_prompt=commentor_system_prompt,
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    )

    review_and_posting_system_prompt = (
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. \n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n   - The review must: \n"
        "   - Be a ~200-300 word review in markdown format. \n"
        "   - Specify what is good about the PR: \n"
        "   - Did the author follow ALL contribution rules? What is missing? \n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n"
        "   - Are there notes on whether new endpoints were documented? \n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
        " If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
        " When you are satisfied, post the review to GitHub.  "
    )

    review_and_posting_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Reviews the drafted comment for completeness and quality, updates the final review in state, and posts it to GitHub.",
        tools=[
            FunctionTool.from_defaults(add_final_review_to_state),
            FunctionTool.from_defaults(post_review_comment)
        ],
        system_prompt=review_and_posting_system_prompt,
        can_handoff_to=["CommentorAgent"],
    )

    return AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": {},
            "draft_comment": "",
            "final_review_comment": "",
        },
    )


def init_llm() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_base = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(model=model, api_key=api_key, api_base=api_base)


# -------------------- GitHub Setup --------------------
class GitHubReviewContext:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_url = os.getenv("REPOSITORY")
        self._validate_env_vars()

        self.git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN")))
        self._repo = None

    def _validate_env_vars(self):
        """Ensure all required environment variables are set."""
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is not set.")
        if not self.repo_url or "github.com" not in self.repo_url:
            raise ValueError("REPO_URL is not set to a valid GitHub repository URL.")

    def get_repo(self):
        if self._repo is not None:
            return self._repo
        if not self.repo_url or "github.com" not in self.repo_url:
            raise RuntimeError("repo_url is not set to a valid GitHub repository URL.")
        owner, repo_name = self._parse_repo_url()
        full_name = f"{owner}/{repo_name}"
        try:
            _repo = self.git.get_repo(full_name)
            return _repo
        except UnknownObjectException as uoe:
            raise RuntimeError(
                f"Repository not found: {full_name}. Ensure the URL and token permissions are correct.") from uoe

    def _parse_repo_url(self) -> tuple[str, str]:
        """Extract (owner, repo) from a GitHub URL like https://github.com/owner/repo(.git)."""
        try:
            parts = self.repo_url.strip().split("github.com/")[-1].split("/")
            owner = parts[0]
            repo = parts[1].removesuffix(".git")
            return owner, repo
        except IndexError as idx_err:
            raise ValueError(f"Invalid GitHub repo URL: {repo_url}") from idx_err

    def close(self):
        if self.git is not None:
            try:
                self.git.close()
            except Exception as e:
                print(f"Error closing GitHub client: {e}")


github = GitHubReviewContext()


# -------------------- Tools (functions) --------------------
async def get_pr_details(pr_number: int) -> Dict[str, Any]:
    """Useful for retrieving details about a specific pull request by its number.
    Returns author (login), title, body, diff_url, state, head_sha, and commit_shas.
    """
    repo = github.get_repo()
    pr = repo.get_pull(pr_number)
    commit_shas: List[str] = [c.sha for c in pr.get_commits()]
    return {
        "user": pr.user.login if pr.user else None,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha if pr.head else None,
        "commit_shas": commit_shas,
    }


def get_file_contents(file_path: str, ref: Optional[str] = None) -> str:
    """Useful for fetching the contents of a file from the repository.
    Provide the file path relative to the repo root. Optionally, specify a ref (branch name or commit SHA).
    """
    repo = github.get_repo()
    file_content = repo.get_contents(file_path, ref=ref) if ref else repo.get_contents(file_path)
    # PyGithub returns ContentFile with base64-encoded content
    return file_content.decoded_content.decode("utf-8")


def get_commit_details(head_sha: str) -> List[Dict[str, Any]]:
    """Useful for retrieving details about files changed in a specific commit given its SHA.
    Returns a list of dicts with filename, status, additions, deletions, changes, and patch.
    """
    repo = github.get_repo()
    commit = repo.get_commit(head_sha)
    changed_files: List[Dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": getattr(f, "patch", None),
        })
    return changed_files


async def add_context_to_state(ctx: Context, key: str, value: Any) -> str:
    """Save the context into the shared workflow state under the given key."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state[key] = value
    return f"Context added to state with key '{key}'."


# -------------------- State management tools --------------------
async def add_review_comment_to_state(ctx: Context, comment: str) -> str:
    """Save a draft review comment into a shared workflow state.

    Parameters:
    - comment: The draft review comment.
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["draft_comment"] = comment
    return "Review comment was successfully added to the state."


# -------------------- Final review + posting tools --------------------
async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Saves the final review comment into the shared workflow state."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["final_review_comment"] = final_review
    return "Final review comment added to state."


def post_review_comment(pr_number: int, comment: str) -> str:
    """Post the final review comment to the specified GitHub pull request as a review.

    Parameters:
    - pr_number: The pull request number.
    - comment: The final review body to post.

    Returns: A short confirmation string containing PR number and review id.
    """
    repo = github.get_repo()
    pr = repo.get_pull(pr_number)
    review = pr.create_review(body=comment)
    try:
        review_id = getattr(review, "id", None)
    except Exception:
        review_id = None
    return f"Posted review to PR #{pr_number}. Review id: {review_id}"


if __name__ == "__main__":
    asyncio.run(main())
    github.close()
