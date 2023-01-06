from starlette.applications import Starlette
from starlette.responses import UJSONResponse
import gpt_2_simple as gpt2
import tensorflow as tf
import uvicorn
import os
import gc
from keybert import KeyBERT

app = Starlette(debug=False)

sess = gpt2.start_tf_sess(threads=6)
gpt2.load_gpt2(sess)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

generate_count = 0
kw_model = KeyBERT()

@app.route('/hashtags', methods=['POST'])
async def homepage(request):
    global kw_model
    if request.method == 'POST':
        params = await request.json()
        dictionary = []
        SingleKeyword=kw_model.extract_keywords(params['caption'], keyphrase_ngram_range=(1, 1), stop_words='english', use_mmr=True, diversity=0.5, top_n=10)
        # Mutiple keywords words
        MutipleLineKeyword=kw_model.extract_keywords(params['caption'], keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.7, top_n=5)
        dictionary.append( {
                'text': params['caption'],
                'SingleKeyword': SingleKeyword,
                'MutipleLineKeyword': MutipleLineKeyword
            })
        print(dictionary)
        gc.collect()
        return UJSONResponse({'data': dictionary},
                            headers=response_header)


@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess

    if request.method == 'POST':
        params = await request.json()
        params = {
            'length': 100,
            'temperature': params['temperature'],
            'top_k': 40,
            'top_p': 0.9,
            'nsamples':10,
            'prefix': params['prefix'],
            'batch_size':10
        }
    text = gpt2.generate(sess,
                         length=100,
                         temperature=float(params['temperature']),
                         prefix=params['prefix'],
                         truncate="_END_",
                         batch_size=params['batch_size'],
                         nsamples=10,
                         return_as_list=True
                         )

    generate_count += 1
    if generate_count == 16:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=6)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    return UJSONResponse({'text': text},
                         headers=response_header)



@app.route('/captions', methods=['POST'])
async def homepage(request):
    global generate_count
    global sess
    global kw_model

    params = await request.json()
    params = {
        'length': params['length'],
        'temperature': params['temperature'],
        'top_k': 40,
        'top_p': 0.9,
        'nsamples':params['output_count'],
        'prefix': params['prefix'],
        'batch_size':params['output_count']
    }
    text = gpt2.generate(sess,
                         length=int(params['length']),
                         temperature=float(params['temperature']),
                         prefix=params['prefix'],
                         truncate="_END_",
                         batch_size=params['batch_size'],
                         nsamples=5,
                         return_as_list=True
                         )

    generate_count += 1
    if generate_count == 16:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=6)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    return UJSONResponse({'data': text},
                         headers=response_header)



if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=int(os.environ.get('PORT', 8080)))


