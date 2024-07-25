import json
import os
from crewai.utilities import Logger

file_path = "crewai_visualization_report.json"


def clear_report(logger: Logger):
  """Clear the report file."""
  if os.path.exists(file_path):
    with open(file_path, "w") as f:
      json.dump({}, f)
      logger.log("info", f"Report file {file_path} cleared.")


def __initialize_report_if_necessary(current_file_path: str):
  """Initialize the report file."""
  if not os.path.exists(current_file_path):
    with open(current_file_path, "w") as f:
      json.dump({}, f)

def register_agent(name, goal, backstory, tools, verbose, allow_delegation, llm_model):
  """Upsert an agent into the report file."""
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
  [register_tool(tool) for tool in tools]

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


def register_tool(tool):
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

def register_step(task_id, task_input, additional_input, role, thought, action, action_input, observation):
  """Upsert a step into the report file."""
  step = {
        "custom_metrics": {},
        "output": {
          "agent": {"agent_id": role},
          "thought": thought,
          "action": action,
          "action_input": action_input,
          "observation": observation
        }
      }
  task_json = {
    "task_id": task_id,
    "input": task_input,
    "additional_input": additional_input,
    "steps": [ step ]
  }

  __initialize_report_if_necessary(file_path)
  with open(file_path, "r+") as f:
    data = json.load(f)

    if "workflow" not in data:
      data["workflow"] = []
    tasks = data["workflow"]
    if not any(task["task_id"] == task_id for task in tasks):
      tasks.append(task_json)
    else:
      # Step needs to be appended to task
      task_index = next(i for i, task in enumerate(tasks) if task["task_id"] == task_id)
      task = tasks[task_index]
      task["steps"].append(step)


    # Move back to the start of the file before writing
    f.seek(0)
    json.dump(data, f, indent=4)
    # Truncate the file to the current position in case new data is shorter than old
    f.truncate()
