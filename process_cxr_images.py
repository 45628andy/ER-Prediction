from google.cloud import storage
import pickle

patient_imgs = pickle.load(open("cxr_imgs.pkl", "rb"))

storage_client = storage.Client("[calm-photon-384805]")

bucket = storage_client.get_bucket("mimic-cxr-jpg-2.0.0.physionet.org")

print(len(patient_imgs))
c = 1
for p_img in patient_imgs:
    print(c)
    blob = bucket.blob(patient_imgs[p_img][1:])
    blob.download_to_filename("./cxr_imgs/" + str(p_img) + ".jpg")
    c+=1

#blob = bucket.blob("files/p10/p10000898/s50771383/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
#blob.download_to_filename("testing.jpg")