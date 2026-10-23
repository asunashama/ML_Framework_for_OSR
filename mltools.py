import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
import numpy as np
import pickle

# Show loss curves
# def show_history(history):
#     plt.figure()
#     plt.title('Training loss performance')
#     plt.plot(history.epoch, history.history['loss'], label='train loss+error')
#     plt.plot(history.epoch, history.history['val_loss'], label='val_error')
#     plt.legend()
#     plt.savefig('figure/total_loss.png')
#     plt.close()
#
#
#     plt.figure()
#     plt.title('Training accuracy performance')
#     plt.plot(history.epoch, history.history['accuracy'], label='train_accuracy')
#     plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
#     plt.legend()
#     plt.savefig('figure/total_acc.png')
#     plt.close()
#
#     train_acc=history.history['accuracy']
#     val_acc=history.history['val_accuracy']
#     train_loss=history.history['loss']
#     val_loss=history.history['val_loss']
#     epoch=history.epoch
#     np_train_acc=np.array(train_acc)
#     np_val_acc=np.array(val_acc)
#     np_train_loss=np.array(train_loss)
#     np_val_loss=np.array(val_loss)
#     np_epoch=np.array(epoch)
#     np.savetxt('train_acc.txt',np_train_acc)
#     np.savetxt('train_loss.txt',np_train_loss)
#     np.savetxt('val_acc.txt',np_val_acc)
#     np.savetxt('val_loss.txt',np_val_loss)

def show_history(accuracy_train,accuracy_valid,save_path='export/accucary.png'):
    iterations = range(len(accuracy_valid))
    plt.figure()
    plt.plot(iterations, accuracy_train, marker='o', linestyle='-', color='r',label='Training Accuracy')
    plt.plot(iterations, accuracy_valid, marker='o', linestyle='-', color='b', label='valid Accuracy')
    plt.title('Accuracy Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_lstm2layer_output(a,modulation_type=None,save_filename=None):
    plt.figure(figsize=(4,3),dpi=600)
    plt.plot(range(128),a[0],label=modulation_type)
    plt.legend()
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])
    plt.savefig(save_filename,dpi=600,bbox_inches ='tight')
    plt.tight_layout()
    plt.close()

def plot_conv4layer_output(a,modulation_type=None):
    plt.figure(figsize=(4,3),dpi=600)
    for i in range(100):
        plt.plot(range(124),a[0,0,:,i])
        plt.xticks([])  #去掉横坐标值
        plt.yticks(size=20)
        save_filename='./figure_conv4_output/output%d.png'%i
        plt.savefig(save_filename,dpi=600,bbox_inches='tight')
        plt.tight_layout()
        plt.close()
 

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    plt.figure(figsize=(8, 6),dpi=600)
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    #plt.title(title,fontsize=10)
    plt.colorbar()
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90,size=12)
    plt.yticks(tick_marks, labels,size=12)
    #np.set_printoptions(precision=2, suppress=True)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i!=j:
                text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10)
            elif i==j:
                if int(np.around(cm[i,j]*100))==100:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=7,color='darkorange')
                else:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10,color='darkorange')


    plt.tight_layout()
    #plt.ylabel('True label',fontdict={'size':8,})
    #plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename,dpi=600,bbox_inches = 'tight')
    #plt.show()
    plt.close()

def calculate_confusion_matrix(Y,Y_hat,classes): # Y是真是标签，Y_hat是预测标签
    n_classes = len(classes)
    conf = np.zeros([n_classes,n_classes])
    confnorm = np.zeros([n_classes,n_classes])

    for k in range(0,Y.shape[0]):
        i = list(Y[k,:]).index(1)
        j = int(np.argmax(Y_hat[k,:]))
        conf[i,j] = conf[i,j] + 1

    for i in range(0,n_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm,right,wrong

def Evaluation_index(answers,result,num_known = 10000,num_unknown = 6500):

    answer_list = list(answers.values())
    result_list = list(result.values())
    count = 0
    for a, b in zip(answer_list, result_list):
        if a == b:
            count += 1
    allclass_accuracy = count / len(answer_list)
    print('\n\nall_accuracy: %.4f' % (allclass_accuracy))
    index = 0
    index2 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    print('\n\n\n\n\n')
    known_labels = [str(i) for i in range(1,21)]
    unknown_labels = ['X%d' % i for i in range(1,14)]
    for item in answer_list:
        index += 1
        if item in known_labels:
            if answer_list[index - 1] == result_list[index - 1]:
                count1 += 1
            if result_list[index - 1] in known_labels:
                count2 += 1
    queshi_accuracy = count2 / num_known
    known_accuracy = count1 / num_known
    print('确识准确率: %.4f' % (queshi_accuracy))
    print('已知类别准确率: %.4f' % (known_accuracy))
    for item in answer_list:
        index2 += 1
        if item in unknown_labels:
            if result_list[index2 - 1] in unknown_labels:
                count3 += 1
            if answer_list[index2 - 1] == result_list[index2 - 1]:
                count4 += 1
    # print(count)
    jushi_accuracy = count3 / num_unknown
    weizhileibie_accuracy = count4 / num_unknown
    print('拒识准确率: %.4f' % (jushi_accuracy))
    print('未知类别准确率: %.4f' % (weizhileibie_accuracy))

    # 计算 Precision / Recall / F1（已知 vs 未知 检测）
    sum_pred_known = sum(1 for r in result_list if r in known_labels)
    sum_pred_unknown = sum(1 for r in result_list if r in unknown_labels)

    TP_known = count2  # 真正例：真实已知且被预测为已知
    FN_known = max(0, num_known - TP_known)
    FP_known = max(0, sum_pred_known - TP_known)
    precision_known = TP_known / (TP_known + FP_known) if (TP_known + FP_known) > 0 else 0.0
    recall_known = TP_known / num_known if num_known > 0 else 0.0
    f1_known = (2 * precision_known * recall_known / (precision_known + recall_known)) if (precision_known + recall_known) > 0 else 0.0

    TP_unknown = count3  # 真正例：真实未知且被预测为未知（拒识）
    FN_unknown = max(0, num_unknown - TP_unknown)
    FP_unknown = max(0, sum_pred_unknown - TP_unknown)
    precision_unknown = TP_unknown / (TP_unknown + FP_unknown) if (TP_unknown + FP_unknown) > 0 else 0.0
    recall_unknown = TP_unknown / num_unknown if num_unknown > 0 else 0.0
    f1_unknown = (2 * precision_unknown * recall_unknown / (precision_unknown + recall_unknown)) if (precision_unknown + recall_unknown) > 0 else 0.0

    print('\n已知检测 (Known) - Precision: %.4f  Recall: %.4f  F1: %.4f' % (precision_known, recall_known, f1_known))
    print('未知检测 (Unknown) - Precision: %.4f  Recall: %.4f  F1: %.4f' % (precision_unknown, recall_unknown, f1_unknown))


def main():
    import rmldataset2016
    import numpy as np
    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        rmldataset2016.load_data()

    one_sample = X_test[0]
    print(np.shape(one_sample))
    print(one_sample[0:2])
    print(np.max(one_sample,axis=1))
    one_sample = np.power(one_sample,2)
    one_sample = np.sqrt(one_sample[0,:]+one_sample[1,:])

    plt.figure()
    plt.title('Training Samples')
    one_sample_t = np.arange(128)
    plt.plot(one_sample_t,one_sample)
    # plt.scatter()
    plt.grid()
    plt.show()

    sum_sample = np.sum(one_sample)
    print(sum_sample)