def IsPalindrome(text):
    if len(text) == 2:
        return [True if text[0] == text[1] else False]
    if len(text) < 2:
        return True

    text = list(text)
    while len(text) > 1:
        if text[0] == text[-1]:
            text.pop(0)
            text.pop(-1)
            IsPalindrome(text)
        else:
            return False

    return True


print(IsPalindrome('aaaaaa'))


def powerset(arr):
    output = [[]]
    for num in arr:
        print(num)
        print(output)
        output += [curr + [num] for curr in output]
    return output


print(powerset([1, 2, 3, 4]))


def all_permutations(arr):
    output = [[]]
    for num in arr:
        output += [curr + [num] for curr in output]

    return output

print(all_permutations([1, 2, 3, 4]))

def differencearray(arr1, arr2):
    response = dashboard.switch.updateDeviceSwitchRoutingInterface(
        serial, interface_id,
        name='L3 interface',
        subnet='192.168.1.0/24',
        interfaceIp='192.168.1.2',
        multicastRouting='disabled',
        vlanId=100,
        ospfSettings={'area': '0', 'cost': 1, 'isPassiveEnabled': True},
        ipv6={"assignmentMode": "static",
              "address": "1:2:3:4::1/48",
              "prefix": "1:2:3:4::/48",
              "gateway": "1:2:3:4::2"}
    )
