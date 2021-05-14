import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from flask import Flask, request, render_template

app = Flask(__name__)

spec_symbols = ['?', '_', '$', '-', '#', '*', '.', '!', '@', '\\', "'", ';', '&', '=',
                '^', '|', '+', '%', ']', '[', '~', '\x03', '`', '/', '(', ')', '}', '>',
                '<', '{', ':', '\x0f', '\x07', '\x02', '\x17']

symbols_total = 34496267

symbols_dict = {'6': 0.026545799868722027,
                '3': 0.030272840826516097,
                '1': 0.053260342633595685,
                'X': 0.0015942014827285515,
                'n': 0.03472720685980312,
                'i': 0.03669060771126337,
                'V': 0.001858693869687407,
                'x': 0.006868018501828038,
                '2': 0.04060259621715011,
                'l': 0.028281871774705362,
                'S': 0.0031432966355461013,
                '5': 0.02849247427265101,
                'I': 0.0022168485650925648,
                'L': 0.0026413292777447486,
                'E': 0.003305430120888153,
                'G': 0.0022771449444080427,
                'T': 0.0026880009944264403,
                '7': 0.027184158796080746,
                '4': 0.026721094198395436,
                'e': 0.04831096651704372,
                's': 0.033269860764934364,
                '8': 0.027772367369489574,
                '9': 0.03426643236498604,
                'a': 0.06122097211272164,
                'u': 0.019128765440040222,
                'r': 0.036823201768469614,
                'h': 0.019120851540255068,
                'o': 0.037048472520229504,
                'p': 0.014677095350636056,
                '0': 0.039856950318711296,
                'm': 0.022858444364429346,
                'z': 0.008543098301042254,
                'j': 0.010769947948280897,
                'N': 0.002612456588418683,
                'Q': 0.0014088191049773588,
                'B': 0.002609992553687041,
                'd': 0.022156426375062554,
                't': 0.028679277093953383,
                'Z': 0.0016599767157414454,
                'P': 0.0023596466249522015,
                'K': 0.0022299804207800223,
                'f': 0.014559836286053793,
                'c': 0.020221376417338142,
                'y': 0.015818900056635114,
                '?': 7.626912210529911e-05,
                'Y': 0.0020363942568046565,
                'k': 0.019639951186602306,
                'g': 0.016458534484325507,
                'w': 0.009145308389455589,
                'b': 0.018378858210947868,
                'O': 0.0021493919907333743,
                'R': 0.0028897909446259796,
                'q': 0.0045707554385522355,
                'J': 0.0019162653164761278,
                'v': 0.01189154756948049,
                'D': 0.0026050934728676584,
                'A': 0.004286028978150012,
                '_': 0.0008498600732653189,
                'M': 0.002765458650931708,
                'U': 0.002105213297427226,
                'H': 0.0021862365571323995,
                'F': 0.002232125580428746,
                'C': 0.002598049232399552,
                'W': 0.0017658142546264498,
                '$': 8.618323831967093e-05,
                '-': 0.00043883009138351114,
                '#': 6.119502727642965e-05,
                '*': 0.0001755552274685258,
                '.': 0.000818117508192988,
                '!': 0.0002209514438185442,
                '@': 0.00016874289615163287,
                '\\': 6.72536538518791e-06,
                "'": 1.5045106184967781e-05,
                ';': 1.2349162302112283e-05,
                '&': 2.2089346653074086e-05,
                '=': 1.8958573111693506e-05,
                '^': 8.667604526599937e-06,
                '|': 1.2755003316735692e-06,
                '+': 5.246944546202637e-06,
                '%': 2.043699395067878e-05,
                ']': 2.29010286823209e-06,
                '[': 2.956841677970547e-06,
                '~': 4.609194380365852e-06,
                '\x03': 7.537047414434727e-07,
                '`': 1.4494321950836014e-06,
                '/': 1.4494321950836014e-06,
                '(': 1.5363981267886175e-06,
                ')': 2.898864390167203e-07,
                '}': 4.348296585250804e-07,
                '>': 8.696593170501608e-08,
                '<': 5.797728780334405e-08,
                '{': 2.898864390167203e-07,
                ':': 2.319091512133762e-07,
                '\x0f': 5.797728780334405e-08,
                '\x07': 2.8988643901672026e-08,
                '\x02': 5.797728780334405e-08,
                '\x17': 2.8988643901672026e-08
                }


def entropy(text):
    entropy = 0
    for i in text:
        entropy += symbols_dict[i] * np.log2(symbols_dict[i])
    return -entropy

def prob_sum(text):
    prob_sum = 0
    for i in text:
        prob_sum += symbols_dict[i]
    return prob_sum

keyboard = {
    '1': (1, 1), '2': (1, 2), '3': (1, 3), '4': (1, 4), '5': (1, 5), '6': (1, 6),
    '7': (1, 7), '8': (1, 8), '9': (1, 9), '0': (1, 10), '-': (1, 11), '=': (1, 12),

    '!': (1, 1), '@': (1, 2), '#': (1, 3), '$': (1, 4), '%': (1, 5), '^': (1, 6),
    '&': (1, 7), '*': (1, 8), '(': (1, 9), ')': (1, 10), '_': (1, 11), '+': (1, 12),

    'q': (2, 1), 'w': (2, 2), 'e': (2, 3), 'r': (2, 4), 't': (2, 5), 'y': (2, 6),
    'u': (2, 7), 'i': (2, 8), 'o': (2, 9), 'p': (2, 10), '[': (2, 11), '{': (2, 11), ']': (2, 12), '}': (2, 12),

    'a': (3, 1), 's': (3, 2), 'd': (3, 3), 'f': (3, 4), 'g': (3, 5), 'h': (3, 6),
    'j': (3, 7), 'k': (3, 8), 'l': (3, 9), ':': (3, 9), ';': (3, 10), '\'': (3, 11), '`': (3, 11), '\\': (3, 12),
    '|': (3, 12),

    '~': (4, 0), 'z': (4, 1), 'x': (4, 2), 'c': (4, 3), 'v': (4, 4), 'b': (4, 5), 'n': (4, 6),
    'm': (4, 7), ',': (4, 8), '<': (4, 8), '.': (4, 9), '>': (4, 9), '/': (4, 10), '?': (4, 10),

    '\x03': (4, 11), '\x0f': (4, 11), '\x07': (4, 11), '\x02': (4, 11), '\x17': (4, 11)}


def keyboard_dist(text):
    tmp = keyboard[text[0].lower()]
    distance = 0
    for i in range(1, len(text)):
        distance += ((keyboard[text[i].lower()][0] - tmp[0]) ** 2 + \
                     (keyboard[text[i].lower()][1] - tmp[1]) ** 2) ** 0.5
        tmp = keyboard[text[i].lower()]
    return distance


def same_line(text):
    tmp = keyboard[text[0].lower()][0]
    for i in range(1, len(text)):
        if keyboard[text[i].lower()][0] == tmp:
            continue
        else:
            return 0
    return 1


def capslock(text):
    for i in text:
        if i.isalpha():
            if i.islower():
                return 0
        else:
            continue
    return 1


vowels = ['a', 'e', 'i', 'o', 'u', 'y']
consonants = ['q', 'w', 'r', 't', 'p', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']


def feature_generator(input_pass):
    X = pd.DataFrame(columns=['Password'])
    X['Password'] = [input_pass]
    # длина пароля и кол-во уникальных символов и доля уникальных символов
    X['length'] = X['Password'].apply(lambda x: len(x))
    X['unique_symbols_cnt'] = X["Password"].apply(lambda x: len(set(x)))
    X['unique_symbols_ratio'] = X['unique_symbols_cnt'] / X['length']
    # кол-во букв в пароле и доля букв в пароле
    X['letters_cnt'] = X['Password'].apply(lambda x: sum([i.isalpha() for i in x]))
    X['letter_ratio'] = X['letters_cnt'] / X['length']
    # кол-во цифр в пароле и доля цифр в пароле
    X['digits_cnt'] = X['Password'].apply(lambda x: sum([i.isdigit() for i in x]))
    X['digits_ratio'] = X['digits_cnt'] / X['length']
    # только цифры/буквы
    X['only_digits'] = X['Password'].apply(lambda x: sum([i.isdigit() for i in x]) == len(x))
    X['only_letters'] = X['Password'].apply(lambda x: sum([i.isalpha() for i in x]) == len(x))
    # кол-во спец символов в пароле и доля спец символов в пароле
    X['spec_symbols_cnt'] = X['Password'].apply(lambda x: sum([1 for i in x if i in spec_symbols]))
    X['spec_symbols_ratio'] = X['spec_symbols_cnt'] / X['length']
    # кол-во букв в верхнем регистре и доля больших букв общая и по буквам
    X['upper_letters_cnt'] = X['Password'].apply(lambda x: sum([c.isupper() for c in x]))
    X['upper_letters_ratio_by_letters'] = X['upper_letters_cnt'] / X['letters_cnt']
    X['upper_letters_ratio_full'] = X['upper_letters_cnt'] / X['length']
    # кол-во букв в верхнем регистре и доля больших букв общая и по буквам
    X['lower_letters_cnt'] = X['Password'].apply(lambda x: sum([c.islower() for c in x]))
    X['lower_letters_ratio_by_letters'] = X['lower_letters_cnt'] / X['letters_cnt']
    X['lower_letters_ratio_full'] = X['lower_letters_cnt'] / X['length']
    # наличие одновременно и нижнего и верхнего регистров
    X['lower_and_upper'] = (X['upper_letters_cnt'] > 0) & (X['lower_letters_cnt']) > 0
    # наличие одновременно и нижнего и верхнего регистров и спец символов
    X['lower_and_upper_and_spec'] = (X['upper_letters_cnt'] > 0) & \
                                    (X['lower_letters_cnt'] > 0) & \
                                    (X['spec_symbols_cnt'] > 0)
    # наличие одновременно и нижнего и верхнего регистров и спец символов и цифр
    X['lower_and_upper_and_spec_and_digits'] = (X['upper_letters_cnt'] > 0) & \
                                               (X['lower_letters_cnt'] > 0) & \
                                               (X['spec_symbols_cnt'] > 0) & \
                                               (X['digits_cnt'] > 0)
    # кол-во уникальных букв
    X['unique_letters_cnt'] = X['Password'].apply(lambda x: len(set([i for i in x if i.isalpha()])))
    # кол-во уникальных цифр
    X['unique_digits_cnt'] = X['Password'].apply(lambda x: len(set([i for i in x if i.isdigit()])))
    # кол-во уникальных спец символов
    X['unique_spec_symbols_cnt'] = X['Password'].apply(lambda x: len(set([i for i in x if i in spec_symbols])))
    # паттерн даты
    X['dates'] = (X['Password'].str.contains('^19[0-9]{2}[0-1][0-9][0-3][0-9]$') & \
                  ~X['Password'].str.contains('^19[0-9]{2}1[3-9][0-3][0-9]$') | \
                  X['Password'].str.contains('^20[0-9]{2}[0-1][0-9][0-3][0-9]$') & \
                  ~X['Password'].str.contains('^20[0-9]{2}1[3-9][0-3][0-9]$'))
    # палиндром и симметрия
    X['palindrom'] = X['Password'] == X['Password'].str[::-1]
    X['symmetric'] = X['Password'].apply(lambda x: x[:len(x) // 2] == x[len(x) // 2:])
    # рейтинг по разнообразию
    X['has_lower_letter'] = X['Password'].apply(
        lambda x: 1 if sum([1 for i in x if i.isalpha() and i.islower()]) > 0 else 0)
    X['has_upper_letter'] = X['Password'].apply(
        lambda x: 1 if sum([1 for i in x if i.isalpha() and i.isupper()]) > 0 else 0)
    X['has_digit'] = X['digits_cnt'].apply(lambda x: 1 if x > 0 else 0)
    X['has_spec_sym'] = X['spec_symbols_cnt'].apply(lambda x: 1 if x > 0 else 0)
    X['complexity_by_content'] = X[['has_lower_letter', 'has_upper_letter', 'has_digit', 'has_spec_sym']].astype(
        'int').sum(axis=1).values
    # частые паттерны
    X['has_pass'] = X['Password'].str.lower().str.contains('pass')
    X['has_passwd'] = X['Password'].str.lower().str.contains('passwd')
    X['has_password'] = X['Password'].str.lower().str.contains('password')
    X['has_123'] = X['Password'].str.contains('123')
    X['has_12345'] = X['Password'].str.contains('12345')
    X['has_321'] = X['Password'].str.contains('321')
    X['has_54321'] = X['Password'].str.contains('54321')
    X['has_1234567890'] = X['Password'].str.contains('1234567890')
    X['has_0987654321'] = X['Password'].str.contains('0987654321')
    X['has_zaq'] = X['Password'].str.contains('zaq')
    X['has_q1w2e3'] = X['Password'].str.contains('q1w2e3')
    X['has_1q2w3e'] = X['Password'].str.contains('1q2w3e')
    X['has_qazwsx'] = X['Password'].str.lower().str.contains('qazwsx')
    X['has_qwerty'] = X['Password'].str.lower().str.contains('qwerty')
    X['has_asdfgh'] = X['Password'].str.lower().str.contains('asdfgh')
    X['has_zxcvbn'] = X['Password'].str.lower().str.contains('zxcvbn')
    # кол-во гласных/согласных
    X['vowel_letters_cnt'] = X['Password'].apply(lambda x: sum([i.isalpha() for i in x if i in vowels]))
    X['consonant_letters_cnt'] = X['Password'].apply(lambda x: sum([i.isalpha() for i in x if i in consonants]))
    # является словом
    # X['word'] = X['Password'].str.lower().apply(lambda x: x in unique_nltk_words)
    # евклидово расстояние на клавиатуре
    X['keyboard_dist'] = X['Password'].apply(lambda x: keyboard_dist(x))
    # на одной строке клавиатуры все символы
    X['same_line'] = X['Password'].apply(lambda x: same_line(x))
    # слово, которое заканчивается
    X['digits_ended'] = X['Password'].str.contains("^[a-zA-Z]+[\d]{1,2}$", regex=True)
    # энтропия и сумма вероятностей и удельная вероятность
    X['entropy'] = X['Password'].apply(lambda x: entropy(x))
    X['prob_sum'] = X['Password'].apply(lambda x: prob_sum(x))
    X['prob_sum_ratio'] = X['prob_sum'] / X['length']
    # capslock
    X['capslock'] = X['Password'].apply(lambda x: capslock(x))
    # подсчет отдельных цифр
    X['countZero'] = X['Password'].apply(lambda x: str(x).count('0'))
    X['countOne'] = X['Password'].apply(lambda x: str(x).count('1'))
    X['countTwo'] = X['Password'].apply(lambda x: str(x).count('2'))
    X['countThree'] = X['Password'].apply(lambda x: str(x).count('3'))
    # подсчет отдельных букв
    X['countA'] = X['Password'].apply(lambda x: str(x).lower().count('a'))
    X['countE'] = X['Password'].apply(lambda x: str(x).lower().count('e'))
    X['countI'] = X['Password'].apply(lambda x: str(x).lower().count('i'))
    X['countO'] = X['Password'].apply(lambda x: str(x).lower().count('o'))
    X['countN'] = X['Password'].apply(lambda x: str(x).lower().count('n'))
    X['countR'] = X['Password'].apply(lambda x: str(x).lower().count('r'))
    X['countS'] = X['Password'].apply(lambda x: str(x).lower().count('s'))

    X['upper_letters_ratio_by_letters'].fillna(0, inplace=True)
    X['lower_letters_ratio_by_letters'].fillna(0, inplace=True)

    X.drop('Password', axis=1, inplace=True)
    return X

model = CatBoostRegressor()
model.load_model('ctb_passwords_model', format='cbm')

def predict(password):
    password = feature_generator(password)
    return np.expm1(model.predict(password)[0])


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = ''
    if request.method == "GET":
        return render_template('index.html')

    else:
        password = request.form['password']
        if not password:
            errors = "Заполни форму"
        else:
            prediction = predict(password)

        if not errors:
            data = {
                'password': password,
                'prediction': prediction,
            }
            return render_template("index.html", password=password, prediction=prediction)

        data = {
            'name': password,
        }

        return render_template("index.html", errors=errors, password=password)


if __name__ == '__main__':
    app.run(debug=True)