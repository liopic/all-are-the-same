import requests
import os
import csv
from bs4 import BeautifulSoup
from config import LIST_URL, TMP_DIR, INFO_CSV, MEMBERS, LEGISLATURA


def retrieve_profile_page(member_id: int, legislatura_id: int):
    r = requests.get(LIST_URL % (member_id, legislatura_id))
    if r.status_code == 200:
        return r.text
    else:
        return ''


def extract_political_information(page_content: str, member_id: int):
    html = BeautifulSoup(page_content, 'html.parser')
    nameTag = html.select('div.nombre_dip')
    name = nameTag[0].text
    groupTag = html.select('p.nombre_grupo')
    group = groupTag[0].text
    twitterTag = html.select('div.webperso_dip_imagen a[href*=twitter]')
    twitter = twitterTag[0].attrs['href'] if len(twitterTag) else ''
    return {'id': member_id, 'name': name, 'group': group, 'twitter': twitter}


def save_member_info_into_csv(members):
    with open(TMP_DIR + '/' + INFO_CSV % (LEGISLATURA), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for member in members:
            csv_writer.writerow([member['id'], member['name'],
                                member['group'], member['twitter']])


if __name__ == "__main__":
    if not os.path.isdir(TMP_DIR):
        os.mkdir(TMP_DIR)

    print(f'Downloading diputados information in {TMP_DIR}')
    members = []
    for member_id in range(MEMBERS):
        page_content = retrieve_profile_page(member_id+1, LEGISLATURA)
        member = extract_political_information(page_content, member_id+1)
        members.append(member)

        print(f"{member['id']}: {member['name']} from {member['group']} {member['twitter']}")

    print(f'Saving information to {TMP_DIR}/{INFO_CSV}')
    save_member_info_into_csv(members)
