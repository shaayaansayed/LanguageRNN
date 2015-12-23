
# Convert the sentences to json
E.g.
python write_to_json.py test.yts2se2e_M80_si_freezecnn_factored_2lstm_0pad_iter_15000_beam_size_1.txt -x

# Evaluate the json file:
E.g
python run_coco_eval.py test.yts2se2e_M80_si_freezecnn_factored_2lstm_0pad_iter_15000_beam_size_1.txt.json youtube_test.json
