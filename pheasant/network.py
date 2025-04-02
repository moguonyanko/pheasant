import dns.resolver

def get_dmarc_record(domain):
    """指定されたドメインのDMARCレコードを取得し、解析する。"""
    try:
        answers = dns.resolver.resolve(domain, 'TXT')
        for rdata in answers:
            text_record = rdata.strings[0].decode('utf-8')
            if text_record.startswith('v=DMARC1;'):
                print(f"DMARCレコードが見つかりました: {text_record}")
                parse_dmarc_record(text_record)
                return
        print(f"{domain} にDMARCレコードは見つかりませんでした。")
    except dns.resolver.NXDOMAIN:
        print(f"{domain} は存在しないドメインです。")
    except dns.resolver.NoAnswer:
        print(f"{domain} にTXTレコードが見つかりませんでした。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def parse_dmarc_record(dmarc_record):
    """DMARCレコードの内容を解析して表示する。"""
    tags = dmarc_record.split(';')
    print("\nDMARCレコードの詳細:")
    for tag in tags:
        tag = tag.strip()
        if tag:
            key, value = tag.split('=', 1)
            print(f"  {key}: {value}")

if __name__ == "__main__":
    domain_to_check = input("DMARCレコードを確認したいドメインを入力してください: ")
    get_dmarc_record(domain_to_check)
