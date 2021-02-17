import typing
from typing import Any, Dict, List, Text, Optional, Type

from transformers import TFBertForTokenClassification, AutoTokenizer, pipeline
from rasa.shared.nlu.constants import ENTITIES
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractor

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased", from_pt=True, model_max_length=512)
model = TFBertForTokenClassification.from_pretrained('monilouise/ner_pt_br', from_pt=True)

nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

class BertEntityExtractor(EntityExtractor):

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        doc = nlp(message.data['text'])
        extracted = self.extract_entities(doc)
        print(extracted)
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)

    @staticmethod
    def extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = []
        for token in doc:
            entities.append(
                {
                    "entity": token['entity_group'],
                    "value": token['word'],
                    "confidence": token['score'],
                }
            )
        return entities