import random


def add_two_digits():
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    sum = num1 + num2
    return num1, num2, sum


result = add_two_digits()
print("两位数相加的结果为：", result[0], "+", result[1], "=", result[2])
