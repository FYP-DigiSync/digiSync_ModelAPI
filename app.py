from starlette.applications import Starlette
from starlette.responses import UJSONResponse
import gpt_2_simple as gpt2
import tensorflow as tf
import uvicorn
import os
import gc

app = Starlette(debug=False)

sess = gpt2.start_tf_sess(threads=1)
gpt2.load_gpt2(sess)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

generate_count = 0


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
            'batch_size':params['batch_size']
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
    if generate_count == 8:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess)
        generate_count = 0

    print(text)
    gc.collect()
    return UJSONResponse({'text': text},
                         headers=response_header)

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=int(os.environ.get('PORT', 8080)))


