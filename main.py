from base64 import encode
from transformers import HerbertTokenizer, RobertaModel

tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1", is_decoder=False)

encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="pt")
print(encoded_input.shape)
outputs = model.generate(encoded_input)

# output = model(
#     **tokenizer.batch_encode_plus(
#         [
#             "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
#             "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
#         ],
#         padding="longest",
#         add_special_tokens=True,
#         return_tensors="pt",
#     )
# )

last_hidden = outputs.last_hidden_state
output_keys = outputs.__dict__.keys()
output_values_not_none = [outputs.__dict__[key] is not None for key in output_keys]
print(outputs.__dict__.keys())
print(output_values_not_none)
print(f"{last_hidden.shape = }")
print(outputs.pooler_output.shape) # this is what will be used for intent classification

# print(last_hidden[0])