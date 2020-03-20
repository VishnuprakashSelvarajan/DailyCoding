# Anagram letters
def anagram(s1, s2):
    # Remove spaces and convert the string to lower
    s1 = s1.replace(' ', '').lower()
    s2 = s2.replace(' ', '').lower()

    # compare the string len if not equal return False
    if len(s1) != len(s2): return False

    count = {}  # Count the frequency of letters

    # Count and add the letters in count
    for letter in s1:
        if letter in count:
            count[letter] += 1
        else:
            count[letter] = 1
    # Do the reverse for S2 and remove from the count
    for letter in s2:
        if letter in count:
            count[letter] -= 1
        else:
            count[letter] = 1

    for k in count:
        if count[k] != 0:
            return False

    return True


print(anagram('Clint Eastwood', 'Old West action'))
