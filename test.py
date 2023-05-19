from gpt4all import GPT4All

gptj = GPT4All("ggml-mpt-7b-chat.bin")


# repl
while True:
    question = input('Ask away: ')
    if question == '\q':
        print("Bye!")
        break
    messages = [{"role": "user", "content": question}]
    gptj.chat_completion(messages)

exit()
