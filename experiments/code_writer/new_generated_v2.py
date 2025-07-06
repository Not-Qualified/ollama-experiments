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
from pydantic import BaseModel, Field
from config import settings
import os
import subprocess
import tempfile
from typing import TypedDict, List, Dict, Any, Annotated, Sequence, Union, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages
from langsmith import Client, traceable
import docker
import git


# Initialize LangSmith client for monitoring
try:
    langsmith_client = Client()
except Exception:
    langsmith_client = None
    print("Warning: LangSmith client not initialized")

# State definition for the development workflow
class DevelopmentState(TypedDict):
    messages: Annotated[List[str], add_messages]
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
    filename: str = Field(description="Name of the file to write")
    content: str = Field(description="Content to write to the file")
    language: str = Field(default="python", description="Programming language")


class AutonomousDeveloperAgent:
    def __init__(self, project_path: str = None):
        self.llm = ChatOllama(
            model=settings.model,
            temperature=0,
        )

        # Initialize Docker client with error handling
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker client not available: {e}")
            self.docker_client = None

        self.project_path = project_path or tempfile.mkdtemp()

        # Ensure project directory exists
        os.makedirs(self.project_path, exist_ok=True)

        self.tools = []
        self.workflow = None

        # Initialize agents as None, will be created in setup_workflow
        self.planning_agent = None
        self.coding_agent = None
        self.testing_agent = None
        self.review_agent = None

        self.setup_tools()
        self.setup_workflow()

    def setup_tools(self):
        """Initialize development tools for the agent"""

        @tool
        def write_code_file(filename: str, content: str, language: str = "python") -> str:
            """Write code to a file with proper formatting"""
            try:
                file_path = os.path.join(self.project_path, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w') as f:
                    f.write(content)

                # Auto-format based on language (with error handling)
                try:
                    if language == "python":
                        result = subprocess.run(["black", file_path],)
                        if result.returncode != 0:
                            print(f"Warning: Black formatting failed for {filename}")
                    elif language == "javascript":
                        result = subprocess.run(["prettier", "--write", file_path], capture_output=True, timeout=30)
                        if result.returncode != 0:
                            print(f"Warning: Prettier formatting failed for {filename}")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    print(f"Warning: Formatter not available for {language}")

                return f"Successfully wrote {filename}"
            except Exception as e:
                return f"Error writing file {filename}: {str(e)}"

        @tool
        def run_tests(test_path: str = "tests/") -> Dict[str, Any]:
            """Run tests and return results"""
            try:
                full_test_path = os.path.join(self.project_path, test_path)

                # Check if test directory exists
                if not os.path.exists(full_test_path):
                    return {"success": False, "error": f"Test path {test_path} does not exist"}

                result = subprocess.run(
                    ["python", "-m", "pytest", full_test_path, "-v"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path,
                    timeout=60
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Test execution timed out"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @tool
        def execute_code_safely(code: str, language: str = "python") -> Dict[str, Any]:
            """Execute code in a safe Docker container"""
            if not self.docker_client:
                return {"success": False, "error": "Docker not available"}

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
                    "output": container.decode('utf-8') if isinstance(container, bytes) else str(container),
                    "errors": ""
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        @tool
        def git_operations(operation: str, message: str = None) -> str:
            """Perform Git operations"""
            try:
                if operation == "init":
                    if not os.path.exists(os.path.join(self.project_path, ".git")):
                        repo = git.Repo.init(self.project_path)
                        return "Git repository initialized"
                    else:
                        return "Git repository already exists"

                # For other operations, ensure repo exists
                try:
                    repo = git.Repo(self.project_path)
                except git.InvalidGitRepositoryError:
                    repo = git.Repo.init(self.project_path)

                if operation == "add":
                    repo.git.add(A=True)
                    return "Files added to staging"

                elif operation == "commit":
                    if repo.is_dirty():
                        repo.index.commit(message or "Automated commit by AI agent")
                        return f"Committed with message: {message or 'Automated commit by AI agent'}"
                    else:
                        return "No changes to commit"

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

                if not os.path.exists(full_path):
                    return {"error": f"File {file_path} does not exist"}

                results = {"lint_issues": "", "type_issues": "", "quality_score": 10}

                # Run linting (with error handling)
                try:
                    lint_result = subprocess.run(
                        ["flake8", full_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    results["lint_issues"] = lint_result.stdout
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    results["lint_issues"] = "Linter not available"

                # Run type checking (with error handling)
                try:
                    type_result = subprocess.run(
                        ["mypy", full_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    results["type_issues"] = type_result.stdout
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    results["type_issues"] = "Type checker not available"

                # Calculate quality score
                issue_count = len([line for line in results["lint_issues"].split('\n') if line.strip()])
                results["quality_score"] = max(0, 10 - issue_count)

                return results
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
        self.planning_agent = self.create_agent(
            "You are a software architect. Break down requirements into detailed development tasks. "
            "Be specific and actionable in your task descriptions."
        )

        self.coding_agent = self.create_agent(
            "You are an expert programmer. Write clean, efficient, well-documented code. "
            "Use the write_code_file tool to create actual files."
        )

        self.testing_agent = self.create_agent(
            "You are a QA engineer. Create comprehensive tests and validate code quality. "
            "Use the run_tests tool to execute tests and write_code_file to create test files."
        )

        self.review_agent = self.create_agent(
            "You are a senior developer doing code review. Ensure code quality and best practices. "
            "Use code_quality_check tool to analyze code quality."
        )

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
            ("system", f"{system_message}\n\nYou have access to the following tools:\n\n{{tools}}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{{tool_names}}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {{input}}\nThought:{{agent_scratchpad}}"),
        ])

        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt,
        )
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=3)

    @traceable(name="planning_node") if langsmith_client else lambda name: lambda func: func
    def planning_node(self, state: DevelopmentState) -> DevelopmentState:
        """Plan the development tasks"""
        try:
            response = self.planning_agent.invoke({
                "input": f"Break down this requirement into specific development tasks: {state['requirements']}\n"
                         f"Consider existing files: {list(state['code_files'].keys())}\n"
                         f"Provide a numbered list of concrete, actionable tasks."
            })

            # Parse tasks from response
            output_text = response.get("output", "")
            tasks = []

            for i, line in enumerate(output_text.split("\n")):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    task_description = line.lstrip("0123456789.- ")
                    if task_description:
                        tasks.append({
                            "id": i,
                            "description": task_description,
                            "status": "pending",
                            "code_files": []
                        })

            if not tasks:
                # Fallback if parsing fails
                tasks = [{
                    "id": 0,
                    "description": state['requirements'],
                    "status": "pending",
                    "code_files": []
                }]

            new_state = state.copy()
            new_state.update({
                "tasks": tasks,
                "current_task_index": 0,
                "messages": state["messages"] + [output_text]
            })

            return new_state

        except Exception as e:
            print(f"Error in planning_node: {e}")
            # Return state with minimal task
            new_state = state.copy()
            new_state.update({
                "tasks": [{"id": 0, "description": state['requirements'], "status": "pending", "code_files": []}],
                "current_task_index": 0,
                "messages": state["messages"] + [f"Planning error: {str(e)}"]
            })
            return new_state

    @traceable(name="coding_node") if langsmith_client else lambda name: lambda func: func
    def coding_node(self, state: DevelopmentState) -> DevelopmentState:
        """Generate code for current task"""
        try:
            if not state["tasks"] or state["current_task_index"] >= len(state["tasks"]):
                return state

            current_task = state["tasks"][state["current_task_index"]]

            response = self.coding_agent.invoke({
                "input": f"Write code for this task: {current_task['description']}\n"
                         f"Use the write_code_file tool to create the necessary files.\n"
                         f"Consider existing files: {list(state['code_files'].keys())}"
            })

            # Update task status
            new_tasks = state["tasks"].copy()
            new_tasks[state["current_task_index"]]["status"] = "coded"

            new_state = state.copy()
            new_state.update({
                "tasks": new_tasks,
                "messages": state["messages"] + [response.get("output", "")]
            })

            return new_state

        except Exception as e:
            print(f"Error in coding_node: {e}")
            new_state = state.copy()
            new_state["messages"] = state["messages"] + [f"Coding error: {str(e)}"]
            return new_state

    @traceable(name="testing_node") if langsmith_client else lambda name: lambda func: func
    def testing_node(self, state: DevelopmentState) -> DevelopmentState:
        """Create and run tests"""
        try:
            if not state["tasks"] or state["current_task_index"] >= len(state["tasks"]):
                return state

            current_task = state["tasks"][state["current_task_index"]]

            response = self.testing_agent.invoke({
                "input": f"Create and run tests for: {current_task['description']}\n"
                         f"Code files: {list(state['code_files'].keys())}\n"
                         f"Use write_code_file to create test files and run_tests to execute them."
            })

            # Update task status
            new_tasks = state["tasks"].copy()
            new_tasks[state["current_task_index"]]["status"] = "tested"

            new_state = state.copy()
            new_state.update({
                "tasks": new_tasks,
                "test_results": {"last_test": "executed"},
                "messages": state["messages"] + [response.get("output", "")]
            })

            return new_state

        except Exception as e:
            print(f"Error in testing_node: {e}")
            new_state = state.copy()
            new_state["messages"] = state["messages"] + [f"Testing error: {str(e)}"]
            return new_state

    @traceable(name="review_node") if langsmith_client else lambda name: lambda func: func
    def review_node(self, state: DevelopmentState) -> DevelopmentState:
        """Review code quality"""
        try:
            if not state["tasks"] or state["current_task_index"] >= len(state["tasks"]):
                return state

            current_task = state["tasks"][state["current_task_index"]]

            response = self.review_agent.invoke({
                "input": f"Review the code quality and completeness for task: {current_task['description']}\n"
                         f"Use code_quality_check tool to analyze any Python files.\n"
                         f"Provide feedback on whether the implementation is complete and ready."
            })

            new_state = state.copy()
            new_state.update({
                "review_feedback": response.get("output", ""),
                "messages": state["messages"] + [response.get("output", "")]
            })

            return new_state

        except Exception as e:
            print(f"Error in review_node: {e}")
            new_state = state.copy()
            new_state.update({
                "review_feedback": f"Review error: {str(e)}",
                "messages": state["messages"] + [f"Review error: {str(e)}"]
            })
            return new_state

    def should_iterate(self, state: DevelopmentState) -> str:
        """Decide whether to iterate, commit, or end"""
        try:
            current_task_index = state["current_task_index"]

            # Check if we need to improve current task
            if "needs improvement" in state.get("review_feedback", "").lower():
                return "iterate"

            # Move to next task if available
            if current_task_index < len(state["tasks"]) - 1:
                # Update task index for next iteration
                state["current_task_index"] = current_task_index + 1
                return "iterate"

            # All tasks completed
            return "commit"

        except Exception as e:
            print(f"Error in should_iterate: {e}")
            return "end"

    def git_commit_node(self, state: DevelopmentState) -> DevelopmentState:
        """Commit code to Git"""
        try:
            commit_message = f"Completed development tasks: {datetime.now().isoformat()}"

            # Initialize git if needed and commit
            git_result = ""
            for tool in self.tools:
                if tool.name == "git_operations":
                    # Initialize git
                    init_result = tool.func("init")
                    # Add files
                    add_result = tool.func("add")
                    # Commit
                    commit_result = tool.func("commit", commit_message)
                    git_result = f"{init_result}; {add_result}; {commit_result}"
                    break

            new_state = state.copy()
            new_state.update({
                "git_status": git_result or "Git operations completed",
                "messages": state["messages"] + [f"Git operations: {git_result}"]
            })

            return new_state

        except Exception as e:
            print(f"Error in git_commit_node: {e}")
            new_state = state.copy()
            new_state.update({
                "git_status": f"Git error: {str(e)}",
                "messages": state["messages"] + [f"Git error: {str(e)}"]
            })
            return new_state

    @traceable(name="autonomous_development") if langsmith_client else lambda name: lambda func: func
    def develop_project(self, requirements: str) -> Dict[str, Any]:
        """Main entry point for autonomous development"""
        try:
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
                "tasks_completed": len(final_state.get("tasks", [])),
                "final_state": final_state,
                "messages": final_state.get("messages", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "project_path": self.project_path,
                "tasks_completed": 0
            }


# Usage Example
def main():
    """Example usage of the Autonomous Developer Agent"""
    try:
        # Initialize the agent
        agent = AutonomousDeveloperAgent(project_path="./demo_project")

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
        print("Starting autonomous development...")
        result = agent.develop_project(requirements)

        print(f"\nDevelopment completed: {result['success']}")
        print(f"Project location: {result['project_path']}")
        print(f"Tasks completed: {result['tasks_completed']}")

        if result.get('error'):
            print(f"Error: {result['error']}")

        if result.get('messages'):
            print("\nExecution log:")
            for i, message in enumerate(result['messages'][-5:]):  # Show last 5 messages
                print(f"{i+1}. {message[:200]}...")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    print("Please run this script as a module using:")
    print("python -m experiments.code_writer.new_generated_v2")
    print("Or install the package in development mode first:")
    print("pip install -e .")
    main()
