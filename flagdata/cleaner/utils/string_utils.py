import re
import stringcase


def hump_turned_to_underline(s):
    snake_case = ''
    for i, char in enumerate(s):
        if char.isupper() and i != 0:
            snake_case += '_'
        snake_case += char.lower()
    return snake_case


def camel_to_snake(name):
    return stringcase.snakecase(name)


def is_camel_case(name):
    """
    检查是否是驼峰命名法。

    :param name: 要检查的字符串。
    :return: 如果是驼峰命名法，返回 True；否则返回 False。
    """
    return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))


if __name__ == '__main__':
    # # 示例用法
    # test_names = ["CamelCase", "camelCase", "snake_case", "AnotherExample", "example"]
    #
    # for name in test_names:
    #     if is_camel_case(name):
    #         print(f"{name} is CamelCase")
    #     else:
    #         print(f"{name} is not CamelCase")
    # 示例用法
    class_names = ["CamelCase", "AnotherExample"]

    for class_name in class_names:
        snake_case_name = camel_to_snake(class_name)
        print(f"{class_name} -> {snake_case_name}")
