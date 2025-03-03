import matplotlib.pyplot as plt
import numpy as np

plt.figure(dpi=1000)
def calculate_eer(fpr, frr):
    
    min_index = np.argmin(np.abs(np.array(fpr) - np.array(frr))) 
    eer = (fpr[min_index] + frr[min_index]) / 2  
    return eer

fpr1 = open("/home/zhangli/CCNet/outputs/data/_hu__fpr.txt")
fpr1l = str(fpr1.readlines())
fpr1 = [float(i)*100 for i in fpr1l[3:-4].split(",")]
frr1 = open("/home/zhangli/CCNet/outputs/data/_hu__frr.txt")
frr1l = str(frr1.readlines())
frr1 = [float(i)*100 for i in frr1l[3:-4].split(",")]

eer1 = calculate_eer(fpr1, frr1)
print(f"EER: {eer1:.4f}%")
#print("len", len(frr1))

fpr2 = open("/home/zhangli/CCNet/outputs/data/_trip__fpr.txt")
frr2 = open("/home/zhangli/CCNet/outputs/data/_trip__frr.txt")
fpr2l = str(fpr2.readlines())
fpr2 = [float(i)*100 for i in fpr2l[3:-4].split(",")]
frr2l = str(frr2.readlines())
frr2 = [float(i)*100 for i in frr2l[3:-4].split(",")]
eer2 = calculate_eer(fpr2, frr2)
print(f"EER: {eer2:.4f}%")
#print("len", len(frr1))

fpr3 = open("/home/zhangli/CCNet/outputs/data/_kl__fpr.txt")
frr3 = open("/home/zhangli/CCNet/outputs/data/_kl__frr.txt")
fpr3l = str(fpr3.readlines())
fpr3 = [float(i)*100 for i in fpr3l[3:-4].split(",")]
frr3l = str(frr3.readlines())
frr3 = [float(i)*100 for i in frr3l[3:-4].split(",")]
eer3 = calculate_eer(fpr3, frr3)
print(f"EER: {eer3:.4f}%")
#print("len", len(frr1))

fpr4 = open("/home/zhangli/CCNet/outputs/data/_bas__fpr.txt")
frr4 = open("/home/zhangli/CCNet/outputs/data/_bas__frr.txt")
fpr4l = str(fpr4.readlines())
fpr4 = [float(i)*100 for i in fpr4l[3:-4].split(",")]
frr4l = str(frr4.readlines())
frr4 = [float(i)*100 for i in frr4l[3:-4].split(",")]
eer4 = calculate_eer(fpr4, frr4)
print(f"EER: {eer4:.4f}%")
#print("len", len(frr1))

fpr5 = open("/home/zhangli/CCNet/outputs/data/_cnn__fpr.txt")
frr5 = open("/home/zhangli/CCNet/outputs/data/_cnn__frr.txt")
fpr5l = str(fpr5.readlines())
fpr5 = [float(i)*100 for i in fpr5l[3:-4].split(",")]
frr5l = str(frr5.readlines())
frr5 = [float(i)*100 for i in frr5l[3:-4].split(",")]
eer5 = calculate_eer(fpr5, frr5)
print(f"EER: {eer5:.4f}%")
#print("len", len(frr1))

fpr6 = open("/home/zhangli/CCNet/outputs/data/_pphu__fpr.txt")
frr6 = open("/home/zhangli/CCNet/outputs/data/_pphu__frr.txt")
fpr6l = str(fpr6.readlines())
fpr6 = [float(i)*100 for i in fpr6l[3:-4].split(",")]
frr6l = str(frr6.readlines())
frr6 = [float(i)*100 for i in frr6l[3:-4].split(",")]
eer6 = calculate_eer(fpr6, frr6)
print(f"EER: {eer6:.4f}%")
#print("len", len(frr1))

fpr7 = open("/home/zhangli/CCNet/outputs/data/_pvhu__fpr.txt")
frr7 = open("/home/zhangli/CCNet/outputs/data/_pvhu__frr.txt")
fpr7l = str(fpr7.readlines())
fpr7 = [float(i)*100 for i in fpr7l[3:-4].split(",")]
frr7l = str(frr7.readlines())
frr7 = [float(i)*100 for i in frr7l[3:-4].split(",")]
eer7 = calculate_eer(fpr7, frr7)
print(f"EER: {eer7:.4f}%")
#print("len", len(frr1))

# plt.plot(fpr7,frr7,label='PV',color='green', linestyle='-.', linewidth=2)
# plt.plot(fpr6,frr6,label='PP',color='green', linestyle='-.', linewidth=2)

#plt.plot(fpr4,frr4,label='SSFEN'+f'(EER={eer4:.4f}%)',color='blue', linewidth=2)
#plt.plot(fpr3,frr3,label='SSFEN+$L_{Sim}$'+f'(EER={eer3:.4f}%)',color='darkorange', linewidth=2)
#
#plt.plot(fpr2,frr2,label='SSFEN+$L_{Sim}$+$L_{Cls}$'+f'(EER={eer2:.4f}%)',color='purple', linewidth=2)
#plt.plot(fpr5,frr5,label='ResNet-18+$L_{Sim}$+$L_{Cls}$+FIFM'+f'(EER={eer5:.4f}%)',color='green', linewidth=2)
#plt.plot(fpr1,frr1,label='SSFEN+$L_{Sim}$+$L_{Cls}$+FIFM'+f'(EER={eer1:.4f}%)',color='red', linewidth=2)

plt.plot(fpr6,frr6,label=f'PolyU_Palmprint(EER={eer6:.4f}%)',color='blue', linewidth=2)
plt.plot(fpr7,frr7,label=f'PolyU_NIR_1(EER={eer7:.4f}%)',color='green', linewidth=2)
plt.plot(fpr1,frr1,label=f'PolyU_Palmprint+PolyU_NIR_1(EER={eer1:.4f}%)',color='red', linewidth=2)

plt.plot([0, 1.5], [0, 1.5], 'k--', lw=1)
plt.xlim([0.0, 1.5])
plt.ylim([0.0, 1.5])
plt.xlabel('False Acceptance Rate (%)')
plt.ylabel('False Rejection Rate (%)')
#   plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc = 'center right')
#   plt.savefig('set113_roc.jpg')
plt.savefig("./Fig14.eps",bbox_inches='tight')
plt.show()
plt.close()
