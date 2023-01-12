import re
input = "hello ?, good-morning! ,it's"
a = re.split(r"[^a-zA-Z0-9\s?!-']", input)
print(a)

         
        

