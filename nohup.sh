path="/home/dell3080ti_01/lx/result/`date +%Y%m%d`"
mkdir -p "${path}"
python="/home/dell3080ti_01/.conda/envs/lx-reid/bin/python -u"
script="/home/dell3080ti_01/lx/ReID-Framework/script"

# 1 Supervised

# 1.1 ResNet-50
# 1.1.1 baseline
# file="${path}/`date +%H%M%S`_supervised_resnet-50.log"
# ${python} ${script}/supervised_resnet50.py -c config/supervised.ini -gpu 0 > ${file} 2>&1 &
# 1.1.2 daon
# file="${path}/`date +%H%M%S`_supervised_resnet-50_daon.log"
# ${python} ${script}/supervised_resnet50_daon.py -c config/supervised_online.ini -gpu 0 > ${file} 2>&1 &
# 1.1.3 daoff
# file="${path}/`date +%H%M%S`_supervised_resnet-50_daoff.log"
# ${python} ${script}/supervised_resnet50_daoff.py -c config/supervised_offline.ini -gpu 0 > ${file} 2>&1 &

# 1.2 Bag of Tricks
# 1.2.1 baseline
# file="${path}/`date +%H%M%S`_supervised_bag.log"
# ${python} ${script}/supervised_bag.py -c config/supervised.ini -gpu 2 > ${file} 2>&1 &
# 1.2.2 daon
# file="${path}/`date +%H%M%S`_supervised_bag_daon.log"
# ${python} ${script}/supervised_bag_daon.py -c config/supervised_online.ini -gpu 3 > ${file} 2>&1 &
# 1.2.3 daoff
# file="${path}/`date +%H%M%S`_supervised_bag_daoff.log"
# ${python} ${script}/supervised_bag_daoff.py -c config/supervised_offline.ini -gpu 3 > ${file} 2>&1 &

# 1.3 AGW
# 1.3.1 baseline
# file="${path}/`date +%H%M%S`_supervised_agw.log"
# ${python} ${script}/supervised_agw.py -c config/supervised.ini -gpu 3 > ${file} 2>&1 &
# 1.3.2 daon
# file="${path}/`date +%H%M%S`_supervised_agw_daon.log"
# ${python} ${script}/supervised_agw_daon.py -c config/supervised_online.ini -gpu 3 > ${file} 2>&1 &
# 1.3.3 daoff
# file="${path}/`date +%H%M%S`_supervised_agw_daoff.log"
# ${python} ${script}/supervised_agw_daoff.py -c config/supervised_offline.ini -gpu 3 > ${file} 2>&1 &

# 2 Unsupervised

# 2.1 ResNet-50
# file="${path}/`date +%H%M%S`_unsupervised_resnet-50.log"
# ${python} ${script}/unsupervised_resnet50.py -c config/unsupervised.ini -gpu 0 > ${file} 2>&1 &

# 2.2 Bag of Tricks
# file="${path}/`date +%H%M%S`_unsupervised_bag.log"
# ${python} ${script}/unsupervised_bag.py -c config/unsupervised.ini -gpu 3 > ${file} 2>&1 &

# 3 Val

# file="${path}/`date +%H%M%S`_val.log"
# ${python} ${script}/val.py -c config/val.ini -gpu 0 > ${file} 2>&1 &

# file="${path}/`date +%H%M%S`_val_da.log"
# ${python} ${script}/val_da.py -c config/val.ini -gpu 0 > ${file} 2>&1 &
