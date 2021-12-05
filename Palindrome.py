def IsPalindrome(text):

    def IsEqual(sub_text):
        if sub_text[0] == sub_text[-1]:
            return 1
        return -1

    start = 0
    end = len(text)-1
    text = list(text)
    while len(text) > 1:
        if IsEqual(text):
            text.pop(start)
            text.pop(end)
            start +=1
            end -=1
        else:
            return False

    return True

print(IsPalindrome('aba'))