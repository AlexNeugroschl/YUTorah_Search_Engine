from datetime import date, timedelta
import pandas as pd
import requests


def generate_calendar(start_date:date=date.today()-timedelta(days=7), end_date:date=date.today()+timedelta(days=3650)):
    df = get_data_from_api(start_date, end_date)
    return map_data(df)

def get_data_from_api(start_date:date, end_date:date) -> pd.DataFrame:
    #uses hebcal api
    url = "https://www.hebcal.com/hebcal"
    params = {
        "v": "1",
        "cfg": "json",
        "maj" : "on",
        "min" : "on",
        "nx" : "on",
        "mf" : "on",
        "ss" : "on",
        "mod" : "on",
        "s" : "on",
        "F" : "on",
        "myomi" : "on",
        "nyomi" : "on",
        "yyomi" : "on",
        "dw" : "on",
        "start": start_date,
        "end": end_date
    }
    response = requests.get(url, params=params).json()["items"]
    return pd.DataFrame(response)

def map_data(df:pd.DataFrame):
    parashat_mapping = {
    'Parashat Bereshit': 'subcategory_Bereishit',
    'Parashat Noach': 'subcategory_Noach',
    'Parashat Lech-Lecha': 'subcategory_Lech Lecha',
    'Parashat Vayera': 'subcategory_Vayeira',
    'Parashat Chayei Sara': 'subcategory_Chayei Sara',
    'Parashat Toldot': 'subcategory_Toldot',
    'Parashat Vayetzei': 'subcategory_Vayeitzei',
    'Parashat Vayishlach': 'subcategory_Vayishlach',
    'Parashat Vayeshev': 'subcategory_Vayeishev',
    'Parashat Miketz': 'subcategory_Mikeitz',
    'Parashat Vayigash': 'subcategory_Vayigash',
    'Parashat Vayechi': 'subcategory_Vayechi',
    'Parashat Shemot': 'subcategory_Shemot',
    'Parashat Vaera': 'subcategory_Va\'era',
    'Parashat Bo': 'subcategory_Bo',
    'Parashat Beshalach': 'subcategory_Beshalach',
    'Parashat Yitro': 'subcategory_Yitro',
    'Parashat Mishpatim': 'subcategory_Mishpatim',
    'Parashat Terumah': 'subcategory_Teruma',
    'Parashat Tetzaveh': 'subcategory_Tetzaveh',
    'Parashat Ki Tisa': 'subcategory_Ki Tisa',
    'Parashat Vayakhel': 'subcategory_Vayakhel',
    'Parashat Pekudei': 'subcategory_Pekudei',
    'Parashat Vayakhel-Pekudei': 'subcategory_Vayakhel-Pekudei',
    'Parashat Vayikra': 'subcategory_Vayikra',
    'Parashat Tzav': 'subcategory_Tzav',
    'Parashat Shmini': 'subcategory_Shmini',
    'Parashat Tazria-Metzora': 'subcategory_Tazria-Metzora',
    'Parashat Tazria': 'subcategory_Tazria',
    'Parashat Metzora': 'subcategory_Metzora',
    'Parashat Achrei Mot-Kedoshim': 'subcategory_Acharei Mot-Kedoshim',
    'Parashat Achrei Mot': 'subcategory_Acharei Mot',
    'Parashat Kedoshim': 'subcategory_Kedoshim',
    'Parashat Emor': 'subcategory_Emor',
    'Parashat Behar-Bechukotai': 'subcategory_Behar-Bechukotai',
    'Parashat Behar': 'subcategory_Behar',
    'Parashat Bechukotai': 'subcategory_Bechukotai',
    'Parashat Bamidbar': 'subcategory_Bamidbar',
    'Parashat Nasso': 'subcategory_Naso',
    'Parashat Beha’alotcha': 'subcategory_Behaalotecha',
    'Parashat Sh’lach': 'subcategory_Shelach',
    'Parashat Korach': 'subcategory_Korach',
    'Parashat Chukat': 'subcategory_Chukat',
    'Parashat Balak': 'subcategory_Balak',
    'Parashat Chukat-Balak': 'subcategory_Chukat-Balak',
    'Parashat Pinchas': 'subcategory_Pinchas',
    'Parashat Matot-Masei': 'subcategory_Matot-Masei',
    'Parashat Matot': 'subcategory_Matot',
    'Parashat Masei': 'subcategory_Masei',
    'Parashat Devarim': 'subcategory_Devarim',
    'Parashat Vaetchanan': 'subcategory_Va\'etchanan',
    'Parashat Eikev': 'subcategory_Eikev',
    'Parashat Re’eh': 'subcategory_Re\'eh',
    'Parashat Shoftim': 'subcategory_Shoftim',
    'Parashat Ki Teitzei': 'subcategory_Ki Teitzei',
    'Parashat Ki Tavo': 'subcategory_Ki Tavo',
    'Parashat Nitzavim': 'subcategory_Nitzavim',
    'Parashat Vayeilech': 'subcategory_Vayeilech',
    'Parashat Nitzavim-Vayeilech': 'subcategory_Nitzavim-Vayeilech',
    'Parashat Ha’azinu': 'subcategory_Haazinu'
}
    nach_mapping = {
    'Joshua': 'subcategory_Yehoshua',
    'Judges': 'subcategory_Shoftim',
    'I Samuel': 'subcategory_Shmuel',
    'II Samuel': 'subcategory_Shmuel',
    'I Kings': 'subcategory_Melachim',
    'II Kings': 'subcategory_Melachim',
    'Isaiah': 'subcategory_Yeshayahu',
    'Jeremiah': 'subcategory_Yirmiyahu',
    'Ezekiel': 'subcategory_Yechezkel',
    'Hosea': 'subcategory_Hosheia',
    'Joel': 'subcategory_Yoel',
    'Amos': 'subcategory_Amos',
    'Obadiah': 'subcategory_Ovadia',
    'Jonah': 'subcategory_Yonah',
    'Micah': 'subcategory_Michah',
    'Nachum': 'subcategory_Nachum',
    'Habakkuk': 'subcategory_Chabakuk',
    'Zephaniah': 'subcategory_Tzefania',
    'Haggai': 'subcategory_Chaggai',
    'Zechariah': 'subcategory_Zecharia',
    'Malachi': 'subcategory_Malachi',
    'Psalms': 'subcategory_Tehillim',
    'Proverbs': 'subcategory_Mishlei',
    'Job': 'subcategory_Iyov',
    'Song of Songs': 'subcategory_Shir Hashirim',
    'Ruth': 'subcategory_Rut',
    'Lamentations': 'subcategory_Eichah',
    'Ecclesiastes': 'subcategory_Kohelet',
    'Esther': 'subcategory_Esther',
    'Daniel': 'subcategory_Daniel',
    'Ezra': 'subcategory_Ezra & Nechemia',
    'Nehemiah': 'subcategory_Ezra & Nechemia',
    'I Chronicles': 'subcategory_Divrei Hayamim',
    'II Chronicles': 'subcategory_Divrei Hayamim'
}
    masechta_mapping = {
    'Berachot': 'subcategory_Berachos',
    'Berakhot': 'subcategory_Berachos',
    'Peah': 'subcategory_Peah',
    'Demai': 'subcategory_Demai',
    'Kilayim': 'subcategory_Kilayim',
    'Sheviit': 'subcategory_Sheviis',
    'Terumot': 'subcategory_Terumos',
    'Maasrot': 'subcategory_Maaseros',
    'Maaser Sheni': 'subcategory_Maaser Sheni',
    'Challah': 'subcategory_Challah',
    'Orlah': 'subcategory_Orla',
    'Bikkurim': 'subcategory_Bikkurim',
    'Shabbat': 'subcategory_Shabbos',
    'Eruvin': 'subcategory_Eruvin',
    'Pesachim': 'subcategory_Pesachim',
    'Shekalim': 'subcategory_Shekalim',
    'Yoma': 'subcategory_Yuma',
    'Sukkah': 'subcategory_Sukkah',
    'Beitzah': 'subcategory_Beitza',
    'Rosh Hashana': 'subcategory_Rosh Hashana',
    'Rosh Hashanah': 'subcategory_Rosh Hashana',
    'Taanit': 'subcategory_Taanis',
    'Megillah': 'subcategory_Megillah',
    'Moed Katan': 'subcategory_Moed Katan',
    'Chagigah': 'subcategory_Chagiga',
    'Yevamot': 'subcategory_Yevamos',
    'Ketubot': 'subcategory_Kesubos',
    'Nedarim': 'subcategory_Nedarim',
    'Nazir': 'subcategory_Nazir',
    'Sotah': 'subcategory_Sotah',
    'Gitin': 'subcategory_Gittin',
    'Gittin': 'subcategory_Gittin',
    'Kiddushin': 'subcategory_Kiddushin',
    'Baba Kamma': 'subcategory_Bava Kamma',
    'Baba Metzia': 'subcategory_Bava Metzia',
    'Baba Batra': 'subcategory_Bava Basra',
    'Bava Kamma': 'subcategory_Bava Kamma',
    'Bava Metzia': 'subcategory_Bava Metzia',
    'Bava Batra': 'subcategory_Bava Basra',
    'Sanhedrin': 'subcategory_Sanhedrin',
    'Makkot': 'subcategory_Makkos',
    'Shevuot': 'subcategory_Shavuos',
    'Avodah Zarah': 'subcategory_Avodah Zara',
    'Horayot': 'subcategory_Horayos',
    'Eduyot': 'subcategory_Eduyot',
    'Avot': 'subcategory_Avot',
    'Zevachim': 'subcategory_Zevachim',
    'Menachot': 'subcategory_Menachot',
    'Chullin': 'subcategory_Chullin',
    'Bechorot': 'subcategory_Bechorot',
    'Bekhorot': 'subcategory_Bechorot',
    'Arachin': 'subcategory_Arachin',
    'Arakhin': 'subcategory_Arachin',
    'Temurah': 'subcategory_Temura',
    'Keritot': 'subcategory_Keritot',
    'Meilah': 'subcategory_Meilah',
    'Kinnim': 'subcategory_Kinnim',
    'Tamid': 'subcategory_Tamid',
    'Midot': 'subcategory_Middot',
    'Middot': 'subcategory_Middot',
    'Kelim': 'subcategory_Keilim',
    'Oholot': 'subcategory_Ohalot',
    'Negaim': 'subcategory_Negaim',
    'Parah': 'subcategory_Parah',
    'Tahorot': 'subcategory_Taharot',
    'Mikvaot': 'subcategory_Mikvaot',
    'Makhshirin': 'subcategory_Machshirin',
    'Zavim': 'subcategory_Zavim',
    'Tevul Yom': 'subcategory_Tevul Yom',
    'Yadayim': 'subcategory_Yadayim',
    'Oktzin': 'subcategory_Ukzin',
    'Niddah': 'subcategory_Nidah'
}
    holiday_mapping = {
    'Tzom Tammuz': 'subcategory_Shiva Asar b\'Tamuz',
    'Shabbat Chazon': 'subcategory_Shabbat Chazon',
    'Tish’a B’Av': 'subcategory_Tisha Bav',
    'Shabbat Nachamu': 'subcategory_Shabbat Nachamu',
    'Tu B’Av': 'subcategory_Tu B\'Av',
    'Leil Selichot': 'subcategory_Slichot',
    'Rosh Hashana': 'subcategory_Rosh Hashana',
    'Rosh Hashana II': 'subcategory_Rosh Hashana',
    'Shabbat Shuva': 'subcategory_Shabbat Shuvah',
    'Tzom Gedaliah': 'subcategory_Tzom Gedalia',
    'Yom Kippur': 'subcategory_Yom Kippur',
    'Sukkot I': 'subcategory_Sukkot',
    'Sukkot II': 'subcategory_Sukkot',
    'Sukkot III (CH’’M)': 'subcategory_Sukkot',
    'Sukkot IV (CH’’M)': 'subcategory_Sukkot',
    'Sukkot V (CH’’M)': 'subcategory_Sukkot',
    'Sukkot VI (CH’’M)': 'subcategory_Sukkot',
    'Sukkot VII (Hoshana Raba)': 'subcategory_Hoshana Rabbah',
    'Shmini Atzeret': 'subcategory_Shmini Atzeret/Simchat Torah',
    'Simchat Torah': 'subcategory_Shmini Atzeret/Simchat Torah',
    'Chanukah: 1 Candle': 'subcategory_Chanukah',
    'Chanukah: 2 Candles': 'subcategory_Chanukah',
    'Chanukah: 3 Candles': 'subcategory_Chanukah',
    'Chanukah: 4 Candles': 'subcategory_Chanukah',
    'Chanukah: 5 Candles': 'subcategory_Chanukah',
    'Chanukah: 6 Candles': 'subcategory_Chanukah',
    'Chanukah: 7 Candles': 'subcategory_Chanukah',
    'Chanukah: 8 Candles': 'subcategory_Chanukah',
    'Chanukah: 8th Day': 'subcategory_Chanukah',
    'Asara B’Tevet': 'subcategory_Assara B\'Tevet',
    'Shabbat Shirah': 'subcategory_Shabbat Shira',
    'Tu BiShvat': 'subcategory_Tu Beshvat',
    'Shabbat Shekalim': 'subcategory_Shekalim',
    'Shabbat Zachor': 'subcategory_Zachor',
    'Ta’anit Esther': 'subcategory_Taanit Esther',
    'Purim': 'subcategory_Purim',
    'Shabbat Parah': 'subcategory_Parah',
    'Shabbat HaChodesh': 'subcategory_Hachodesh',
    'Shabbat HaGadol': 'subcategory_Shabbat Hagadol',
    'Pesach I': 'subcategory_Pesach',
    'Pesach II': 'subcategory_Pesach',
    'Pesach III (CH’’M)': 'subcategory_Pesach',
    'Pesach IV (CH’’M)': 'subcategory_Pesach',
    'Pesach V (CH’’M)': 'subcategory_Pesach',
    'Pesach VI (CH’’M)': 'subcategory_Pesach',
    'Pesach VII': 'subcategory_Pesach',
    'Pesach VIII': 'subcategory_Pesach',
    'Yom HaShoah': 'subcategory_Yom Hashoah',
    'Yom HaZikaron': 'subcategory_Yom Hazikaron',
    'Yom HaAtzma’ut': 'subcategory_Yom Ha\'atzmaut',
    'Pesach Sheni': 'subcategory_Pesach Sheni',
    'Lag BaOmer': 'subcategory_Lag BaOmer',
    'Yom Yerushalayim': 'subcategory_Yom Yerushalayim',
    'Shavuot I': 'subcategory_Shavuot',
    'Shavuot II': 'subcategory_Shavuot',
    'Tu B’Av, Shabbat Nachamu': 'subcategory_Tu B\'Av',
    'Tu BiShvat, Shabbat Shirah': 'subcategory_Tu Beshvat',
    'Purim Katan': 'subcategory_Purim Katan',
    'Tish’a B’Av (observed)': 'subcategory_Tisha Bav'
}
    rosh_chodesh_mapping = {
    'Rosh Chodesh Av': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Elul': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Cheshvan': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Kislev': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Tevet': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Sh’vat': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Adar': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Nisan': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Iyyar': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Sivan': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Tamuz': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Adar I': 'subcategory_Rosh Chodesh',
    'Rosh Chodesh Adar II': 'subcategory_Rosh Chodesh'
}
    df = df[["title", "date", "category"]]
    values_to_drop = ["Yom HaAliyah", "Rosh Hashana LaBehemot", "Erev Tish’a B’Av", "Erev Rosh Hashana", "Erev Yom Kippur", "Erev Sukkot", "Sigd", "Chag HaBanot", "Erev Purim", "Purim Meshulash", "Erev Pesach", "Shushan Purim", "Shushan Purim Katan", "Erev Shavuot", "Ta’anit Bechorot", "Tu B’Av"]
    df = df[~((df['category'] == 'holiday') & (df['title'].isin(values_to_drop)))]
    df = df.pivot_table(index="date", columns="category", values="title", aggfunc=lambda x: ', '.join(x))
    df.reset_index(inplace=True)
    df.drop(columns=['mevarchim'], inplace=True)
    #parashat processing
    df['parashat'] = df['parashat'].replace(parashat_mapping)
    df['parashat'] = df['parashat'].bfill()
    #nachyomi processing
    df[['n_sefer', 'n_num']] = df['nachyomi'].str.extract(r'([A-Za-z\s]+?)\s*(\d+)', expand=True)
    df.drop(columns=['nachyomi'], inplace=True)
    df['n_sefer'] = df['n_sefer'].replace(nach_mapping)
    #dafyomi processing
    df[['d_masechta', 'd_num']] = df['dafyomi'].str.extract(r'([A-Za-z\s]+?)\s*(\d+)', expand=True)
    df.drop(columns=['dafyomi'], inplace=True)
    df['d_masechta'] = df['d_masechta'].replace(masechta_mapping)
    #dafWeekly processing
    df[['dw_masechta', 'dw_num']] = df['dafWeekly'].str.extract(r'([A-Za-z\s]+?)\s*(\d+)', expand=True)
    df.drop(columns=['dafWeekly'], inplace=True)
    df['dw_masechta'] = df['dw_masechta'].replace(masechta_mapping)
    df['dw_masechta'] = df['dw_masechta'].ffill()
    df['dw_num'] = df['dw_num'].ffill()
    #yerushalmiyomi processing
    df[['y_masechta', 'y_num']] = df['yerushalmi'].str.extract(r'([A-Za-z\s]+?)\s*(\d+)', expand=True)
    df.drop(columns=['yerushalmi'], inplace=True)
    df['y_masechta'] = df['y_masechta'].replace(masechta_mapping)
    #mishnayomi processing
    df[['m_masechta', 'm_num1', 'm_num2']] = df['mishnayomi'].str.extract(r'([A-Za-z\s]+?)\s*(\d+):(\d+)', expand=True)
    df.drop(columns=['mishnayomi'], inplace=True)
    df['m_masechta'] = df['m_masechta'].replace(masechta_mapping)
    #holiday processing
    df['holiday'] = df['holiday'].replace(r'Rosh Hashana \d+', 'Rosh Hashana', regex=True)
    df['holiday'] = df['holiday'].replace(holiday_mapping)
    #roshchodesh processing
    df['roshchodesh'] = df['roshchodesh'].replace(rosh_chodesh_mapping)
    return df