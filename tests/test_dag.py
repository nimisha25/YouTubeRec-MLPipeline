import pytest
from airflow.models import DagBag

@pytest.fixture
def dagbag():
    """
    Fixture to load the DAGs from DagBag.
    """
    return DagBag()

def test_dag_loaded(dagbag):
    """
    Tests if the 'youtube_etl_pipeline' DAG is loaded.
    """
    dag = dagbag.get_dag("youtube_etl_pipeline")
    assert dag is not None, "DAG 'youtube_etl_pipeline' failed to load."

def test_task_count(dagbag):
    """
    Tests if the 'youtube_etl_pipeline' DAG has 7 tasks.
    """
    dag = dagbag.get_dag("youtube_etl_pipeline")
    assert len(dag.tasks) == 7, "DAG 'youtube_etl_pipeline' does not have 7 tasks."

def test_task_dependencies(dagbag):
    """
    Tests task dependencies in 'youtube_etl_pipeline'.
    """
    dag = dagbag.get_dag("youtube_etl_pipeline")
    task_dependencies = {
        "extract_data": ["transform_data"],
        "transform_data": ["clean_data"],
        "clean_data": ["load_data"]
    }
    for upstream_task, downstream_tasks in task_dependencies.items():
        upstream = dag.get_task(upstream_task)
        downstream_list = [task.task_id for task in upstream.downstream_list]
        for downstream_task in downstream_tasks:
            assert downstream_task in downstream_list, (
                f"Task '{downstream_task}' is not downstream of '{upstream_task}'"
            )
