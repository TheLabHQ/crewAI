import json
import os
from dataclasses import dataclass, asdict
from typing import Union, Optional, List

file_path = "crewai_visualization_report.json"
artifact_directory = "crewai_artifacts"

def clear_report() -> Union[str, None]:
  """Clear the report file and return the name of the report file."""
  if os.path.exists(file_path):
    with open(file_path, "w") as f:
      json.dump({}, f)
    return file_path
  return None

def clear_artifacts() -> Union[str, None]:
  """Clear the artifact directory and return the name of the directory."""
  if os.path.exists(artifact_directory):
    for artifact_filename in os.listdir(artifact_directory):
      os.remove(os.path.join(artifact_directory, artifact_filename))
    return artifact_directory
  return None


def __initialize_report_if_necessary(current_file_path: str):
  """Initialize the report file."""
  if not os.path.exists(current_file_path):
    with open(current_file_path, "w") as f:
      json.dump({}, f)

def register_agent(agent):
  """Upsert an agent into the report file."""

  name = agent.role
  goal = agent.goal
  backstory = agent.backstory
  tools = agent.tools
  verbose = agent.verbose
  allow_delegation = agent.allow_delegation
  llm_model = agent.llm.model_name

  agent_json = {
    "agent_id": name,
    "name": name,
    "goal": goal,
    "backstory": backstory,
    "tools": [{"tool_id": tool.name, "name": tool.name, "description": tool.description} for tool in tools],
    "config": {
      "verbose": verbose,
      "allow_delegation": allow_delegation,
      "llm_model": llm_model
    }
  }
  [_register_tool(tool) for tool in tools]

  __initialize_report_if_necessary(file_path)
  with open(file_path, "r+") as f:
    data = json.load(f)

    if "agents" not in data:
      data["agents"] = []
    agents = data["agents"]
    if not any(agent["name"] == name for agent in agents):
      agents.append(agent_json)
    else:
      # Agent needs to be overwritten
      agent_index = next(i for i, agent in enumerate(agents) if agent["name"] == name)
      agents[agent_index] = agent_json

    # Move back to the start of the file before writing
    f.seek(0)
    json.dump(data, f, indent=4)
    # Truncate the file to the current position in case new data is shorter than old
    f.truncate()


def _register_tool(tool):
  """Upsert tools into the report file."""

  __initialize_report_if_necessary(file_path)
  with open(file_path, "r+") as f:
    data = json.load(f)
    if "tools" not in data:
      data["tools"] = []
    tools = data["tools"]
    if not any(t["name"] == tool.name for t in tools):
      tools.append({"tool_id": tool.name, "name": tool.name, "description": tool.description})
    # Move back to the start of the file before writing
    f.seek(0)
    json.dump(data, f, indent=4)
    # Truncate the file to the current position in case new data is shorter than old
    f.truncate()


def register_answer_step(
        parent_step_id: Optional[str],
        step_id: str,
        task_id,
        task_input,
        additional_input,
        role,
        thought,
        answer
):
  """Upsert an answer step into the report file."""
  step_json = {
        "step_id": step_id,
        "custom_metrics": {},
        "output": {
          "agent": {"agent_id": role},
          "thought": thought,
          "type": "answer",
          "content": {
            "answer": answer
          }
        }
      }
  task_json = {
    "task_id": task_id,
    "parent_step_id": parent_step_id,
    "input": task_input,
    "additional_input": additional_input,
    "steps": [ step_json ],
    "artifacts": []
  }

  _register_step(step_json, step_id, task_id, task_json)

def register_toolcall_step(
        parent_step_id: Optional[str],
        step_id: str,
        task_id,
        task_input,
        additional_input,
        role,
        thought,
        action,
        action_input,
        observation,
):
  """Upsert a toolcall_step into the report file."""
  action_input_dict = action_input
  if isinstance(action_input, str):
    try:
      action_input_dict = json.loads(action_input)
    except json.JSONDecodeError:
      pass

  artifacts = _collect_artifacts(step_id)
  artifacts_asdict = list(map(lambda artifact: asdict(artifact), artifacts))

  step_json = {
        "step_id": step_id,
        "custom_metrics": {},
        "output": {
          "agent": {"agent_id": role},
          "thought": thought,
          "type": "tool-call",
          "content": {
            "action": action,
            "action_input": action_input_dict,
            "observation": observation,
            "artifacts": artifacts_asdict
          }
        }
      }
  task_json = {
    "task_id": task_id,
    "parent_step_id": parent_step_id,
    "input": task_input,
    "additional_input": additional_input,
    "steps": [ step_json ],
    "artifacts": artifacts_asdict
  }

  _register_step(step_json, step_id, task_id, task_json)


def _register_step(step_json, step_id, task_id, task_json):
  __initialize_report_if_necessary(file_path)
  with open(file_path, "r+") as f:
    data = json.load(f)

    if "workflow" not in data:
      data["workflow"] = []
    tasks = data["workflow"]
    if not any(task["task_id"] == task_id for task in tasks):
      tasks.append(task_json)
    else:
      task_index = next(i for i, task in enumerate(tasks) if task["task_id"] == task_id)
      task = tasks[task_index]

      if any(step_dict["step_id"] == step_id for step_dict in task["steps"]):
        # Step needs to be overwritten
        step_index = next(i for i, step_dict in enumerate(task["steps"]) if step_dict["step_id"] == step_json["step_id"])
        task["steps"][step_index] = step_json
      else:
        # Step needs to be appended
        task["steps"].append(step_json)

      accumulated_artifacts = []
      for step_dict in task["steps"]:
        if step_dict["output"]["type"] == "tool-call":
          step_artifacts = step_dict["output"]["content"]["artifacts"]
          accumulated_artifacts = accumulated_artifacts + step_artifacts
      task["artifacts"] = accumulated_artifacts

    # Move back to the start of the file before writing
    f.seek(0)
    json.dump(data, f, indent=4)
    # Truncate the file to the current position in case new data is shorter than old
    f.truncate()

@dataclass
class Artifact:
  artifact_id: str
  file_name: str
  relative_path: str

def _collect_artifacts(step_id: str) -> List[Artifact]:
  """Parse artifacts from the artifact directory."""
  artifacts: List[Artifact] = []
  if os.path.exists(artifact_directory):
    for artifact_filename in os.listdir(artifact_directory):
      if artifact_filename.startswith(step_id):
        artifact = Artifact(
          artifact_id=artifact_filename,
          file_name=artifact_filename,
          relative_path=os.path.join(artifact_directory, artifact_filename)
        )
        artifacts.append(artifact)
  return artifacts