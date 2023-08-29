import re
from tqdm import tqdm
revised_dict = {}
initials_list = ['ng', 'zh', 'ch', 'sh', 'rh', 'b', 'p', 'm', 'f',
                 'v', 'd', 't', 'n', 'l', 'z', 'c', 's', 'j', 'q', 'x', 'g', 'k', 'h']
head_vowel_list = ['i', 'u']
main_vowel_list = ['er', 'ii', 'a', 'e', 'i', 'o', 'u']
end_vowel_list = ['ng', 'b', 'd', 'g', 'm', 'n', 'i', 'u']
no_main_list = ['ng', 'm', 'n']
accent_list = ['24', '11', '31', '55', '2', '5']
initials_list = '|'.join(initials_list).upper()
head_vowel_list = '|'.join(head_vowel_list).upper()
main_vowel_list = '|'.join(main_vowel_list).upper()
end_vowel_list = '|'.join(end_vowel_list).upper()
accent_list = '|'.join(accent_list).upper()
no_main_list = '|'.join(no_main_list).upper()


def hakka_parse(pinyin: str):
    global initials_list, head_vowel_list, main_vowel_list, end_vowel_list, accent_list
    result = []
    pinyin = pinyin.upper()
    no_main_vowel = re.search(f'^(?:{no_main_list})(?:{accent_list})$', pinyin)
    one_vowel = re.search(
        f'^(?:{initials_list})?(?:{main_vowel_list})(?:{accent_list})$', pinyin)
    two_vowel_end = re.search(
        f'^(?:{initials_list})?(?:{main_vowel_list})(?:{end_vowel_list})(?:{accent_list})$', pinyin)
    two_vowel_head = re.search(
        f'^(?:{initials_list})?(?:{head_vowel_list})(?:{main_vowel_list})(?:{accent_list})$', pinyin)
    three_vowel = re.search(
        f'^(?:{initials_list})?(?:{head_vowel_list})(?:{main_vowel_list})(?:{end_vowel_list})(?:{accent_list})$', pinyin)
    if (no_main_vowel):
        result = re.split(rf"({accent_list})", pinyin)
        result = list(filter(lambda x: x != '', result))
    elif (one_vowel):
        result = re.split(rf"((?:{initials_list})|(?:{accent_list}))", pinyin)
        result = list(filter(lambda x: x != '', result))
    elif (two_vowel_end):
        result = re.split(rf"((?:{initials_list})|(?:{accent_list}))", pinyin)
        result = list(filter(lambda x: x != '', result))
        if (result[0] in initials_list):  # with initials
            vowels = re.split(rf"({main_vowel_list})",
                              ''.join(result[1:-1]), maxsplit=1)
            vowels = list(filter(lambda x: x != '', vowels))
            result = [result[0]] + vowels + [result[-1]]
        else:
            vowels = re.split(rf"({main_vowel_list})",
                              ''.join(result[0:-1]), maxsplit=1)
            vowels = list(filter(lambda x: x != '', vowels))
            result = vowels + [result[-1]]
    elif (two_vowel_head):
        result = re.split(rf"((?:{initials_list})|(?:{accent_list}))", pinyin)
        result = list(filter(lambda x: x != '', result))
        if (result[0] in initials_list):  # with initials
            vowels = re.split(rf"({head_vowel_list})",
                              ''.join(result[1:-1]), maxsplit=1)
            vowels = list(filter(lambda x: x != '', vowels))
            result = [result[0]] + vowels + [result[-1]]
        else:
            vowels = re.split(rf"({head_vowel_list})",
                              ''.join(result[0:-1]), maxsplit=1)
            vowels = list(filter(lambda x: x != '', vowels))
            result = vowels + [result[-1]]
    elif (three_vowel):
        result = re.split(rf"((?:{initials_list})|(?:{accent_list}))", pinyin)
        result = list(filter(lambda x: x != '', result))
        if (result[0] in initials_list):  # with initials
            vowels = re.split(rf"({head_vowel_list})",
                              ''.join(result[1:-1]), maxsplit=1) # 1st / 2nd&3rd vowel
            vowels = list(filter(lambda x: x != '', vowels))
            head_vowel = vowels[0]
            other_vowels = re.split(
                rf"({main_vowel_list})", vowels[1], maxsplit=1) # 2nd / 3rd vowel
            other_vowels = list(filter(lambda x: x != '', other_vowels))
            result = [result[0], head_vowel] + other_vowels + [result[-1]]
        else:
            vowels = re.split(rf"({head_vowel_list})",
                              ''.join(result[0:-1]), maxsplit=1) # 1st / 2nd&3rd vowel
            vowels = list(filter(lambda x: x != '', vowels))
            head_vowel = vowels[0]
            other_vowels = re.split(
                rf"({main_vowel_list})", vowels[1], maxsplit=1) # 2nd / 3rd vowel
            other_vowels = list(filter(lambda x: x != '', other_vowels))
            result = [head_vowel] + other_vowels + [result[-1]]
    if('iii' in pinyin):
        return ["<UNK>"]
    if(len(result)>0):
        return result
    return ["<UNK>"]