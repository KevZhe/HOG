from HOG import HOG
import re

def write_features(files, path):
    #write db feature values to txt file
    for image in files:
        #compute feature values
        vals = HOG("data/" + image, normalized = False)
        #fetch file name
        img_name = re.split("/|\.", image)[1]
        #write feature values to file
        with open(path + img_name + ".txt", 'w') as f:
            for line in vals:
                f.write(str(line) + '\n')
            f.close()

def main():
    #images to comute features on
    db_images = ["DB_Images_Pos/DB2.bmp", "DB_Images_Pos/DB9.bmp", "DB_Images_Neg/DB15.bmp"]
    test_images = ["Test_Images_Pos/T2.bmp", "Test_Images_Pos/T5.bmp", "Test_Images_Neg/T10.bmp"]

    #write feature values to txt
    write_features(db_images, "results/DB/")
    write_features(test_images, "results/Test/")

    
if __name__ == "__main__":
    main()