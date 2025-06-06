"""
Autonomous Developer AI Agent
Built with LangChain, LangGraph, and LangSmith

This implementation creates a self-autonomous agent capable of:
- Understanding requirements
- Planning development tasks
- Writing code
- Running tests
- Performing code reviews
- Managing Git operations
"""
from langchain_core.agents import AgentAction, AgentFinish
from pydantic import BaseModel
import os
import subprocess
import tempfile
from typing import TypedDict, List, Dict, Any, Annotated, Sequence, Union
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent, AgentOutputParser
from langchain.tools import Tool, tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langsmith import Client, traceable
import docker
import git


# Initialize LangSmith client for monitoring
langsmith_client = Client()

# State definition for the development workflow
class DevelopmentState(TypedDict):
    messages: Annotated[list, add_messages]
    requirements: str
    tasks: List[Dict[str, Any]]
    current_task_index: int
    code_files: Dict[str, str]
    test_results: Dict[str, Any]
    review_feedback: str
    git_status: str
    deployment_status: str
    iteration_count: int
    project_path: str


class WriteCodeFileInput(BaseModel):
    filename: str
    content: str


class WriteCodeFileParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        return PydanticOutputParser(pydantic_object=WriteCodeFileInput).parse(text)




class AutonomousDeveloperAgent:
    def __init__(self, project_path: str = None):
        self.testing_agent = None
        self.tools: Sequence[Tool] = []
        self.workflow = None
        self.llm = ChatOllama(
            model="codellama:13b",
            temperature=0,
            #num_ctx=10240,
        )
        self.docker_client = docker.from_env()
        self.project_path = project_path or tempfile.mkdtemp()
        self.setup_tools()
        self.setup_workflow()

    def setup_tools(self):
        """Initialize development tools for the agent"""

        @tool
        def write_code_file(filename: str, content: str, language: str = "python") -> DevelopmentState:
            """Write code to a file with proper formatting"""
            try:
                file_path = os.path.join(self.project_path, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w') as f:
                    f.write(content)

                # Auto-format based on language
                if language == "python":
                    subprocess.run(["black", file_path], capture_output=True)
                elif language == "javascript":
                    subprocess.run(["prettier", "--write", file_path], capture_output=True)

                return DevelopmentState(
                    messages=[],
                    requirements="",
                    tasks=[],
                    current_task_index=0,
                    code_files={filename: content},
                    test_results={},
                    review_feedback="",
                    git_status="",
                    deployment_status="",
                    iteration_count=0,
                    project_path=self.project_path
                )
            except Exception as e:
                return {"review_feedback": str(e)}
        @tool
        def run_tests(test_path: str = "tests/") -> Dict[str, Any]:
            """Run tests and return results"""
            try:
                full_test_path = os.path.join(self.project_path, test_path)
                result = subprocess.run(
                    ["python", "-m", "pytest", full_test_path, "-v", "--json-report"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        @tool
        def execute_code_safely(code: str, language: str = "python") -> Dict[str, Any]:
            """Execute code in a safe Docker container"""
            try:
                if language == "python":
                    image = "python:3.11-slim"
                    cmd = ["python", "-c", code]
                elif language == "javascript":
                    image = "node:18-slim"
                    cmd = ["node", "-e", code]
                else:
                    return {"success": False, "error": f"Unsupported language: {language}"}

                container = self.docker_client.containers.run(
                    image,
                    cmd,
                    remove=True,
                    capture_output=True,
                    timeout=30
                )

                return {
                    "success": True,
                    "output": container.decode('utf-8'),
                    "errors": ""
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        @tool
        def git_operations(operation: str, message: str = None) -> str:
            """Perform Git operations"""
            try:
                repo = git.Repo(self.project_path)

                if operation == "init":
                    if not os.path.exists(os.path.join(self.project_path, ".git")):
                        repo = git.Repo.init(self.project_path)
                    return "Git repository initialized"

                elif operation == "add":
                    repo.git.add(A=True)
                    return "Files added to staging"

                elif operation == "commit":
                    repo.index.commit(message or "Automated commit by AI agent")
                    return f"Committed with message: {message}"

                elif operation == "status":
                    return str(repo.git.status())

                else:
                    return f"Unknown git operation: {operation}"

            except Exception as e:
                return f"Git error: {str(e)}"

        @tool
        def code_quality_check(file_path: str) -> Dict[str, Any]:
            """Perform code quality analysis"""
            try:
                full_path = os.path.join(self.project_path, file_path)

                # Run linting
                lint_result = subprocess.run(
                    ["flake8", full_path],
                    capture_output=True,
                    text=True
                )

                # Run type checking
                type_result = subprocess.run(
                    ["mypy", full_path],
                    capture_output=True,
                    text=True
                )

                return {
                    "lint_issues": lint_result.stdout,
                    "type_issues": type_result.stdout,
                    "quality_score": 10 - len(lint_result.stdout.split('\n'))
                }
            except Exception as e:
                return {"error": str(e)}

        self.tools = [
            write_code_file,
            run_tests,
            execute_code_safely,
            git_operations,
            code_quality_check
        ]

    def setup_workflow(self):
        """Setup the LangGraph workflow"""

        # Create specialized agents for different phases
        self.planning_agent = self.create_agent("""You are a software architect. Break down requirements into detailed development tasks.
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}""")

        self.coding_agent = self.create_agent("""You are an expert programmer. Write clean, efficient, well-documented code.
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}""")

        self.testing_agent = self.create_agent("""You are a QA engineer. Create comprehensive tests and validate code quality.
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}""")

        self.review_agent = self.create_agent("""You are a senior developer doing code review. Ensure code quality and best practices.
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}""")

        # Create the workflow graph
        workflow = StateGraph(DevelopmentState)

        # Add nodes
        workflow.add_node("planning", self.planning_node)
        workflow.add_node("coding", self.coding_node)
        workflow.add_node("testing", self.testing_node)
        workflow.add_node("review", self.review_node)
        workflow.add_node("git_commit", self.git_commit_node)

        # Add edges
        workflow.add_edge("planning", "coding")
        workflow.add_edge("coding", "testing")
        workflow.add_edge("testing", "review")

        # Conditional edge for iteration
        workflow.add_conditional_edges(
            "review",
            self.should_iterate,
            {
                "iterate": "coding",
                "commit": "git_commit",
                "end": END
            }
        )

        workflow.add_edge("git_commit", END)
        workflow.set_entry_point("planning")

        self.workflow = workflow.compile()

    def create_agent(self, system_message: str) -> AgentExecutor:
        """Create a specialized agent with tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            # ("placeholder", "{agent_scratchpad}"),
            # ("tools", "{tools}"),
            # ("tool_names", "{tool_names}"),
        ])

        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt,
            # WriteCodeFileParser()
        )
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    @traceable(name="planning_node")
    def planning_node(self, state: DevelopmentState) -> DevelopmentState:
        """Plan the development tasks"""
        response = self.planning_agent.invoke({
            "input": f"Write code for this task: {state['requirements']}\n"
                     f"Consider existing files: {list(state['code_files'].keys())}",
            "content": "Your generated code content here"
        })

        # Parse tasks from response (simplified - in practice, use structured output)
        tasks = [
            {"id": i, "description": task.strip(), "status": "pending", "code_files": []}
            for i, task in enumerate(response["output"].split("\n"))
            if task.strip() and not task.startswith("-")
        ]

        return DevelopmentState(
            **state,
            tasks=tasks,
            current_task_index=0,
            messages=[response["output"]]
        )

    @traceable(name="coding_node")
    def coding_node(self, state: DevelopmentState) -> DevelopmentState:
        """Generate code for current task"""
        current_task = state["tasks"][state["current_task_index"]]

        response = self.coding_agent.invoke({
            "input": f"Write code for this task: {current_task['description']}\n"
                     f"Consider existing files: {list(state['code_files'].keys())}"
        })

        # Update task status
        state["tasks"][state["current_task_index"]]["status"] = "coded"

        return DevelopmentState(
            **state,
            messages=state["messages"] + [response["output"]],
        )

    @traceable(name="testing_node")
    def testing_node(self, state: DevelopmentState) -> DevelopmentState:
        """Create and run tests"""
        current_task = state["tasks"][state["current_task_index"]]

        response = self.testing_agent.invoke({
            "input": f"Create tests for: {current_task['description']}\n"
                     f"Code files: {state['code_files']}"
        })

        # Update task status
        state["tasks"][state["current_task_index"]]["status"] = "tested"

        return DevelopmentState(
            **state,
            test_results={"last_test": "passed"},  # Simplified
            messages=["messages"] + [response["output"]]
        )

    @traceable(name="review_node")
    def review_node(self, state: DevelopmentState) -> DevelopmentState:
        """Review code quality"""
        response = self.review_agent.invoke({
            "input": f"Review the code quality and completeness for task: "
                     f"{state['tasks'][state['current_task_index']]['description']}"
        })

        return DevelopmentState(
            **state,
            review_feedback=response["output"],
            messages=state["messages"] + [response["output"]]
        )

    def should_iterate(self, state: DevelopmentState) -> str:
        """Decide whether to iterate, commit, or end"""
        # Simple logic - in practice, use LLM to decide
        current_task_index = state["current_task_index"]

        if "needs improvement" in state.get("review_feedback", "").lower():
            return "iterate"
        elif current_task_index < len(state["tasks"]) - 1:
            state["current_task_index"] += 1
            return "iterate"
        else:
            return "commit"

    def git_commit_node(self, state: DevelopmentState) -> DevelopmentState:
        """Commit code to Git"""
        commit_message = f"Completed development tasks: {datetime.now().isoformat()}"

        # Use git tool
        git_result = None
        for tool in self.tools:
            if tool.name == "git_operations":
                git_result = tool.func("commit", commit_message)
                break

        return DevelopmentState(
            **state,
            git_status=git_result or "Committed successfully"
        )

    @traceable(name="autonomous_development")
    def develop_project(self, requirements: str) -> Dict[str, Any]:
        """Main entry point for autonomous development"""

        initial_state = DevelopmentState(
            messages=[],
            requirements=requirements,
            tasks=[],
            current_task_index=0,
            code_files={},
            test_results={},
            review_feedback="",
            git_status="",
            deployment_status="",
            iteration_count=0,
            project_path=self.project_path
        )

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)

        return {
            "success": True,
            "project_path": self.project_path,
            "tasks_completed": len(final_state["tasks"]),
            "final_state": final_state
        }


# Usage Example
def main():
    """Example usage of the Autonomous Developer Agent"""

    # Initialize the agent
    agent = AutonomousDeveloperAgent(project_path="./demo_directory")

    # Define requirements
    requirements = """
    Create a Python web API using FastAPI that:
    1. Has a user authentication system
    2. Manages a simple todo list
    3. Includes CRUD operations for todos
    4. Has proper error handling
    5. Includes unit tests
    6. Uses SQLite database
    """

    # Run autonomous development
    result = agent.develop_project(requirements)

    print(f"Development completed: {result['success']}")
    print(f"Project location: {result['project_path']}")
    print(f"Tasks completed: {result['tasks_completed']}")


if __name__ == "__main__":
    main()