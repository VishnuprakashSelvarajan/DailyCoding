"""Linked Lists"""
"""
Efficiency of Linked List Operations:

Operation                               Singly-Linked List  Doubly-Linked List

Access an element                       O(n)                O(n)
Add/remove at an iterator position.     O(1)                O(1)
Add/remove first element.               O(1)                O(1)
Add last element.                       O(1)                O(1)
Remove last element.                    O(n)                O(1)
"""

class node:
    def __init__(self, data=None):
        self.data=data
        self.next=None

class linked_list:
    def __init__(self):
        self.head = node()

    def append(self,data):
        new_node = node(data)
        cur = self.head
        while cur.next != None:
            cur = cur.next
        cur.next = new_node

    def length(self):
        cur = self.head
        total = 0
        while cur.next != None:
            total += 1
            cur = cur.next
        return total

    def display(self):
        elements = []
        cur_node = self.head
        while cur_node.next != None:
            cur_node = cur_node.next
            elements.append(cur_node.data)
        print(elements)

    def get(self,index):
        if index >= self.length():
            print("ERROR: 'get' out of range")
            return None
        cur_index = 0
        cur_node = self.head
        while True:
            cur_node = cur_node.next
            if cur_index == index:  return cur_node.data
            cur_index += 1

    def delete(self, index):
        if index >= self.length():
            print("ERROR: 'get' out of range")
            return None
        cur_index = 0
        cur_node = self.head
        while True:
            last_node = cur_node
            cur_node = cur_node.next
            if cur_index == index:
                last_node.next = cur_node.next
                return
            cur_index += 1




my_list = linked_list()
my_list.display()
my_list.append(1)
my_list.append(2)
my_list.append(3)
my_list.append(4)
my_list.append(5)
my_list.display()
print("The Element on the 3rd Index is %s" % my_list.get(3))
my_list.delete(3)
my_list.display()

