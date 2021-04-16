import typing
from typing import Any, Dict, List, Text, Optional, Type

from transformers import TFBertForSequenceClassification, BERTTokenizer
from rasa.shared.nlu.constants import INTENT
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractor

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', model_max_length = 128, do_lower_case=False)
model = TFBertForSequenceClassification.from_pretrained('./experimento_1')

class BertIntentClassifier(IntentClassifier):

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        intent = self.classify_intent(message.data['text'])
        message.set(
            INTENT, intent, add_to_output=True
        )

    @staticmethod
    def classify_intent(doc: "Doc") -> Dict[Text, Any]:
        outputs = model(tokenizer(doc, return_tensors="tf"))
        intent_indice = tf.argmax(tf.nn.softmax(
            outputs['logits'], axis=-1
        ), axis = -1)
        confidence = tf.nn.softmax(
            outputs['logits'], axis=-1
        )[intent_indice]
        intent = model.config.id2label[intent_indice]
        return {"name": intent, "confidence": confidence}