from celery import Celery, Task
from celery.utils.log import get_task_logger
from lab_tools.settings import settings


logger = get_task_logger(__name__)


class LabToolsTask(Task):

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        pass

    def on_success(self, retval, task_id, args, kwargs):
        pass


app = Celery('lab_tools',
             broker=settings.celery_broker_url,
             broker_connection_retry_on_startup=True,
             task_cls=LabToolsTask,
             include=['lab_tools.analysis.tasks'])
