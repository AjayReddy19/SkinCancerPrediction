# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def runhome():

    return render_template("home.html")


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.


@app.route("/showresult", methods=["GET", "POST"])
def show():
    pic = request.files["pic"]
    inputimg = Image.open(pic)
    inputimg = inputimg.resize((28, 28))
    img = np.array(inputimg).reshape(-1, 28, 28, 3)
    result = SCD.model.predict(img)

    result = result.tolist()
    print(result)
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)
    print(class_ind)
    result = SCD.classes[class_ind]

    if class_ind == 0:
        info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion. If you have several actinic keratoses, your health care provider might prescribe a medicated cream or gel to remove them, such as fluorouracil (Carac, Efudex others), imiquimod (Aldara, Zyclara) or diclofenac. These products might cause inflamed skin, scaling or a burning sensation for a few weeks."

    elif class_ind == 1:
        info = "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck. To cure this, it requires various surgeries like Curettage and electrodesiccation, Mohs Surgery etc"
    elif class_ind == 2:
        info = "Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women. The pathogenesis of BLK is unclear; however, it has been suggested that BLK may be associated with the inflammatory stage of regressing solar lentigo (SL)1. Silicone gel may be used to protect the lichenoid keratosis site from irritating clothing friction. This gel creates a protective barrier to help keep the area moisturized and reduce itching."
    elif class_ind == 3:
        info = "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. They are firm and often feel like a stone under the skin. No treatment is usually necessary for dermatofibromas. Simple reassurance that the lesion is benign may be indicated, unless one of the aggressive subtypes is suspected or diagnosed. Intralesional steroid injections have been attempted with variable results"
    elif class_ind == 4:
        info = "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with ‘melanocytic nevus’, but there are also sources that equate the term mole with any nevus form. The majority of melanocytic nevi do not require treatment. Nevi that are benign but bothersome to the patient because of appearance or irritation can be removed. Moles are removed by the shave technique or suture surgery."
    elif class_ind == 5:
        info = "Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color. They tend to bleed because they contain a large number of blood vessels. They’re also known as lobular capillary hemangioma or granuloma telangiectaticum. Generally, the only way to cure pyogenic granuloma is removing the lesion."
    elif class_ind == 6:
        info = "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat. The exact cause of all melanomas isn't clear, but exposure to ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases your risk of developing melanoma. Melanoma treatment often starts with surgery to remove the cancer. Other treatments may include radiation therapy and treatment with medicine. Treatment for melanoma depends on several factors. These factors include the stage of your cancer, your overall health and your own preferences."

    return render_template("reults.html", result=result,info=info)


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
