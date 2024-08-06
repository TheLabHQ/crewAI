class ObservabilityConfig:
  """singleton class to make the observability configuration changeable"""
  def __init__(self):
    self.report_directory = "crewai_visualization_reports"
    self.artifact_directory = "crewai_artifacts"


observability_config = ObservabilityConfig()
