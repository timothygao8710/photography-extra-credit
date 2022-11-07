def run(image_name, to_display = False):
    
    #Set-Up
    import numpy as np 
    import cv2
    
    proto_path = 'models/colorization_deploy_v2.prototxt'
    model_path = 'models/colorization_release_v2.caffemodel'
    kernal_path = 'models/pts_in_hull.npy'

    #Initialize Model
    net = cv2.dnn.readNetFromCaffe(proto_path,model_path)
    points = np.load(kernal_path)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    points = points.transpose().reshape(2,313,1,1)

    net.getLayer(class8).blobs = [points.astype("float32")]

    temp = np.zeros((1,313), dtype="float32")
    for i in range(313):
        temp[0][i] = 2.606

    net.getLayer(conv8).blobs = [temp]

    #Process Image
    test_image = cv2.imread('input_images/' + image_name)
    n = test_image.shape[1]
    m = test_image.shape[0]
    
    normalized = test_image.astype("float32")/255.0
    lab = cv2.cvtColor(normalized,cv2.COLOR_BGR2LAB)

    #model takes in 224x224
    resized = cv2.resize(lab,(224,224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1,2,0))

    #Convert Back
    ab = cv2.resize(ab, (n,m))

    L = cv2.split(lab)[0]
    res = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

    res = cv2.cvtColor(res,cv2.COLOR_LAB2BGR)
    res = np.clip(res,0,1)

    res = (255 * res).astype("uint8")

    #Display
    if to_display:
        cv2.imshow("Original Image", test_image)
        cv2.imshow("Colorized Image", res)
        cv2.waitKey(0)
    return res

# run(image_name = 'Aldous-Huxley-246x300.jpeg', to_display = True)