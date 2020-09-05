# Scraping data
LIST_URL = ("http://www.congreso.es/portal/page/portal/Congreso/Congreso/"
            "Diputados/BusqForm?_piref73_1333155_73_1333154_1333154.next_page"
            "=/wc/fichaDiputado?idDiputado=%d&idLegislatura=%d")
IMG_URL = "http://www.congreso.es/wc/htdocs/web/img/diputados/%d_%d.jpg"
TMP_DIR = "/tmp/diputados"
INFO_CSV = "members_%d.csv"

LEGISLATURA = 13
MEMBERS = 354  # Legislatura 12 = 393, Legislatura 13 = 354

# Neural network hyperparams
FILTER_SIZES = [96, 256, 512, 768]
KERNEL_SIZE = 5
EPOCHS = 50
BATCH_SIZE = 64

GENERATED_IMAGES_DIR = "generated_images"
