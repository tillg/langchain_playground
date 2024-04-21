from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from rag.utils.dict2file import write_dict_to_file
from rag.utils.utils import get_now_as_string
import os
LOG_DIR = 'data/logs'

class HandlerToDocumentLog(BaseCallbackHandler):

    def create_filename(self, event_name='something'):
        """Create the filename for the log file."""
        filename = os.path.join(LOG_DIR, f'{get_now_as_string()}_{event_name}.log')
        return filename


    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        filename = self.create_filename('chain_start')
        event_doc = {"serialized": serialized, "inputs": inputs}
        write_dict_to_file(dictionary=event_doc, full_filename=filename)

