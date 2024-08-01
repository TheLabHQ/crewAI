import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Union, Optional, List

from langchain_openai import ChatOpenAI

report_directory = "crewai_visualization_reports"
report_filename_base = "crewai_visualization_report"
artifact_directory = "crewai_artifacts"
base_prompt = "Summarize the following with one short english sentence "

llm = ChatOpenAI(model="gpt-4o")

def __get_full_report_file_path() -> str:
  return os.path.join(report_directory, report_filename_base + "_full.json")


def __get_incremented_step_report_file_path() -> str:
  if not os.path.exists(report_directory):
    raise Exception(f"report directory {report_directory} does not exist")
  replay_path_base = os.path.join(report_directory, report_filename_base + "_replay")
  replay_report_paths = glob.glob(replay_path_base + "*")
  return replay_path_base + "__step_" + str(len(replay_report_paths)).zfill(6) + ".json"


def clear_report() -> str:
  """Clear the report directory, initialize reports and return the name of the report directory."""
  if os.path.exists(report_directory):
    for report_filename in os.listdir(report_directory):
      os.remove(os.path.join(report_directory, report_filename))
  else:
    os.mkdir(report_directory)

  with open(__get_full_report_file_path(), "w") as f:
    json.dump({}, f)
  with open(__get_incremented_step_report_file_path(), "w") as f:
    json.dump({}, f)

  return report_directory


def clear_artifacts() -> Union[str, None]:
  """Clear the artifact directory and return the name of the directory."""
  if os.path.exists(artifact_directory):
    for artifact_filename in os.listdir(artifact_directory):
      os.remove(os.path.join(artifact_directory, artifact_filename))
    return artifact_directory
  return None


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

  full_report_file_path = __get_full_report_file_path()

  with open(full_report_file_path, "r") as f:
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

  current_step_report_file_path = __get_incremented_step_report_file_path()
  for file_path in [full_report_file_path, current_step_report_file_path]:
    with open(file_path, "w") as f:
      json.dump(data, f, indent=4)


def _register_tool(tool):
  """Upsert tools into the report file."""

  full_report_file_path = __get_full_report_file_path()
  with open(full_report_file_path, "r") as f:
    data = json.load(f)
    if "tools" not in data:
      data["tools"] = []
    tools = data["tools"]
    if not any(t["name"] == tool.name for t in tools):
      tools.append({"tool_id": tool.name, "name": tool.name, "description": tool.description})

  current_step_report_file_path = __get_incremented_step_report_file_path()
  for file_path in [full_report_file_path, current_step_report_file_path]:
    with open(file_path, "w") as f:
      json.dump(data, f, indent=4)


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
  thought = llm.invoke(f"{base_prompt} from a first person perspective: \n\n{thought}").content
  answer = llm.invoke(f"{base_prompt}: \n\n{answer}").content
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
  task_json = generate_task_json(additional_input, [], parent_step_id, step_json, task_id, task_input)
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
  observation
):
  """Upsert a toolcall_step into the report file."""
  action_input_dict = action_input
  if isinstance(action_input, str):
    try:
      action_input_dict = json.loads(action_input)
      if "context" in action_input_dict:
        action_input_dict["context"] = llm.invoke(f"{base_prompt} as a command: \n\n{action_input_dict['context']}").content
    except json.JSONDecodeError:
      pass

  artifacts = _collect_artifacts(step_id)
  artifacts_asdict = list(map(lambda artifact: asdict(artifact), artifacts))

  if thought:
    thought = llm.invoke(f"{base_prompt} from a first person perspective: \n\n{thought}").content
  if observation:
    observation = llm.invoke(
      f"You are an agent and you have run the function '{action}' "
      f"with the arguments '{json.dumps(action_input_dict)}' .\n\n"
      f"What follows is the result of the function call. "
      f"{base_prompt} from a first person perspective: \n\n{observation}"
    ).content

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
  task_json = generate_task_json(additional_input, artifacts_asdict, parent_step_id, step_json, task_id, task_input)
  _register_step(step_json, step_id, task_id, task_json)


def generate_task_json(additional_input, artifacts_asdict, parent_step_id, step_json, task_id, task_input):
  task_input = llm.invoke(f"{base_prompt}: \n\n{task_input}").content
  task_json = {
    "task_id": task_id,
    "parent_step_id": parent_step_id,
    "input": task_input,
    "additional_input": additional_input,
    "steps": [step_json],
    "artifacts": artifacts_asdict if artifacts_asdict else []
  }
  return task_json


def _register_step(step_json, step_id, task_id, task_json):
  full_report_file_path = __get_full_report_file_path()
  with open(full_report_file_path, "r") as f:
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
        step_index = next(
          i for i, step_dict in enumerate(task["steps"]) if step_dict["step_id"] == step_json["step_id"])
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

  current_step_report_file_path = __get_incremented_step_report_file_path()
  for file_path in [full_report_file_path, current_step_report_file_path]:
    with open(file_path, "w") as f:
      json.dump(data, f, indent=4)


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
