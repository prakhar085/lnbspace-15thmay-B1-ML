from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections

pairs = (
    (r'i need (.*)',  (
            'Sure, i will provide %1',
            'why do you need %1'
        )),
    (r'i want (.*)',  (
            'you should get %1',
            'if you work hard, you will get %1'
        )),
    (r'will you please (.*)',  (
            'well, sure!',
            'no i won\'t',
            'if you feel so, i would surely'
        ))
)

cb = Chat(pairs, reflections)

app  = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def homepage():
    resp = ''
    if request.method == 'POST':
        req = request.form['data']
        resp = cb.respond(req)
    return '''
        <html>
        <body>
            <h3>Chatbot Test</h3>
            <form method = "post>
                <input type = "text" name = "data">
                <input type = "submit"value = "Send">
            </form>
            <p>{}</p>
        </body>
        </html>
        '''.format(resp)
        
@app.route('/chatbot', methods = ['GET', 'POST'])
def chatbot_page():
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run()
        