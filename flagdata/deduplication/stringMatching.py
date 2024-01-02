# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import List

"""
https://leetcode.cn/problems/string-matching-in-an-array/solutions/1723228/shu-zu-zhong-de-zi-fu-chuan-pi-pei-by-le-rpmt/
"""


def stringMatching(words: List[str]) -> List[str]:
    ans = []
    for i, x in enumerate(words):
        for j, y in enumerate(words):
            if j != i and x in y:
                ans.append(x)
                break
    return ans


if __name__ == '__main__':
    list = ["mass", "as", "hero", "superhero"]
    list2 = ["leetcode", "et", "code"]
    list3 = ["blue", "green", "bu"]
    print(stringMatching(list))
    print(stringMatching(list2))
    print(stringMatching(list3))
