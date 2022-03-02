gpu='0'
w= 224
h= 224
epoch= 10
epoch_freeze=0
bs= 32
cp= None
lr= 0
tm= 'part'
pretrained= 1
catch= 1
mname='efficientnet_b0'
print_freq=2
thre=0.5
tta=1.
mpath=required
calculate=1
test='data/test'
yaml='data/all_labels.yaml'
if_feature=1

train:
	python ./train.py \
		--gpu ${gpu} \
		--shape ${w} ${h} \
		--mname ${mname} \
		--bs ${bs} \
		--epoch ${epoch} \
		--epoch_freeze ${epoch_freeze} \
		--cp ${cp} \
		--lr ${lr} \
		--tm ${tm} \
		--pretrained ${pretrained} \
		--catch ${catch} \
		--thre ${thre} \
		--print_freq ${print_freq}
		
	
	
test:
	python ./test.py \
		--gpu ${gpu} \
		--shape ${w} ${h} \
		--mname ${mname} \
		--mpath ${mpath} \
		--calculate ${calculate} \
		--test ${test} \
		--thre ${thre} \
		--yaml ${yaml} \
		--tta ${tta} \
		--if_feature ${if_feature}
		
		