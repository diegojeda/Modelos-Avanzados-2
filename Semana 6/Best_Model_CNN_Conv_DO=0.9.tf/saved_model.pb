эъ
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ех	

conv2d_174/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_174/kernel

%conv2d_174/kernel/Read/ReadVariableOpReadVariableOpconv2d_174/kernel*&
_output_shapes
: *
dtype0
v
conv2d_174/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_174/bias
o
#conv2d_174/bias/Read/ReadVariableOpReadVariableOpconv2d_174/bias*
_output_shapes
: *
dtype0

conv2d_175/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *"
shared_nameconv2d_175/kernel

%conv2d_175/kernel/Read/ReadVariableOpReadVariableOpconv2d_175/kernel*&
_output_shapes
:  *
dtype0
v
conv2d_175/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_175/bias
o
#conv2d_175/bias/Read/ReadVariableOpReadVariableOpconv2d_175/bias*
_output_shapes
: *
dtype0

conv2d_176/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_176/kernel

%conv2d_176/kernel/Read/ReadVariableOpReadVariableOpconv2d_176/kernel*&
_output_shapes
: @*
dtype0
v
conv2d_176/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_176/bias
o
#conv2d_176/bias/Read/ReadVariableOpReadVariableOpconv2d_176/bias*
_output_shapes
:@*
dtype0
}
dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*!
shared_namedense_114/kernel
v
$dense_114/kernel/Read/ReadVariableOpReadVariableOpdense_114/kernel*
_output_shapes
:	Р@*
dtype0
t
dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_114/bias
m
"dense_114/bias/Read/ReadVariableOpReadVariableOpdense_114/bias*
_output_shapes
:@*
dtype0
|
dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_115/kernel
u
$dense_115/kernel/Read/ReadVariableOpReadVariableOpdense_115/kernel*
_output_shapes

:@*
dtype0
t
dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_115/bias
m
"dense_115/bias/Read/ReadVariableOpReadVariableOpdense_115/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_174/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_174/kernel/m

,Adam/conv2d_174/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_174/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_174/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_174/bias/m
}
*Adam/conv2d_174/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_174/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_175/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameAdam/conv2d_175/kernel/m

,Adam/conv2d_175/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_175/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_175/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_175/bias/m
}
*Adam/conv2d_175/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_175/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_176/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_176/kernel/m

,Adam/conv2d_176/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_176/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_176/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_176/bias/m
}
*Adam/conv2d_176/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_176/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_114/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*(
shared_nameAdam/dense_114/kernel/m

+Adam/dense_114/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/m*
_output_shapes
:	Р@*
dtype0

Adam/dense_114/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_114/bias/m
{
)Adam/dense_114/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_115/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_115/kernel/m

+Adam/dense_115/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_115/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/m
{
)Adam/dense_115/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_174/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_174/kernel/v

,Adam/conv2d_174/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_174/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_174/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_174/bias/v
}
*Adam/conv2d_174/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_174/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_175/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameAdam/conv2d_175/kernel/v

,Adam/conv2d_175/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_175/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_175/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_175/bias/v
}
*Adam/conv2d_175/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_175/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_176/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_176/kernel/v

,Adam/conv2d_176/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_176/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_176/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_176/bias/v
}
*Adam/conv2d_176/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_176/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_114/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*(
shared_nameAdam/dense_114/kernel/v

+Adam/dense_114/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/kernel/v*
_output_shapes
:	Р@*
dtype0

Adam/dense_114/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_114/bias/v
{
)Adam/dense_114/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_114/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_115/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_115/kernel/v

+Adam/dense_115/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_115/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_115/bias/v
{
)Adam/dense_115/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_115/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*йI
valueЯIBЬI BХI

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
R
:	variables
;regularization_losses
<trainable_variables
=	keras_api
R
>	variables
?regularization_losses
@trainable_variables
A	keras_api
h

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
R
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemЈmЉ"mЊ#mЋ0mЌ1m­BmЎCmЏLmАMmБvВvГ"vД#vЕ0vЖ1vЗBvИCvЙLvКMvЛ
 
F
0
1
"2
#3
04
15
B6
C7
L8
M9
F
0
1
"2
#3
04
15
B6
C7
L8
M9
­
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
regularization_losses
Znon_trainable_variables

[layers
trainable_variables
	variables
 
][
VARIABLE_VALUEconv2d_174/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_174/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
\metrics
	variables
]layer_metrics
^layer_regularization_losses
regularization_losses

_layers
trainable_variables
`non_trainable_variables
 
 
 
­
ametrics
	variables
blayer_metrics
clayer_regularization_losses
regularization_losses

dlayers
trainable_variables
enon_trainable_variables
 
 
 
­
fmetrics
	variables
glayer_metrics
hlayer_regularization_losses
regularization_losses

ilayers
 trainable_variables
jnon_trainable_variables
][
VARIABLE_VALUEconv2d_175/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_175/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
kmetrics
$	variables
llayer_metrics
mlayer_regularization_losses
%regularization_losses

nlayers
&trainable_variables
onon_trainable_variables
 
 
 
­
pmetrics
(	variables
qlayer_metrics
rlayer_regularization_losses
)regularization_losses

slayers
*trainable_variables
tnon_trainable_variables
 
 
 
­
umetrics
,	variables
vlayer_metrics
wlayer_regularization_losses
-regularization_losses

xlayers
.trainable_variables
ynon_trainable_variables
][
VARIABLE_VALUEconv2d_176/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_176/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
­
zmetrics
2	variables
{layer_metrics
|layer_regularization_losses
3regularization_losses

}layers
4trainable_variables
~non_trainable_variables
 
 
 
Б
metrics
6	variables
layer_metrics
 layer_regularization_losses
7regularization_losses
layers
8trainable_variables
non_trainable_variables
 
 
 
В
metrics
:	variables
layer_metrics
 layer_regularization_losses
;regularization_losses
layers
<trainable_variables
non_trainable_variables
 
 
 
В
metrics
>	variables
layer_metrics
 layer_regularization_losses
?regularization_losses
layers
@trainable_variables
non_trainable_variables
\Z
VARIABLE_VALUEdense_114/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_114/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
В
metrics
D	variables
layer_metrics
 layer_regularization_losses
Eregularization_losses
layers
Ftrainable_variables
non_trainable_variables
 
 
 
В
metrics
H	variables
layer_metrics
 layer_regularization_losses
Iregularization_losses
layers
Jtrainable_variables
non_trainable_variables
\Z
VARIABLE_VALUEdense_115/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_115/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
В
metrics
N	variables
layer_metrics
 layer_regularization_losses
Oregularization_losses
layers
Ptrainable_variables
non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

 count
Ё	variables
Ђ	keras_api
I

Ѓtotal

Єcount
Ѕ
_fn_kwargs
І	variables
Ї	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
 1

Ё	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ѓ0
Є1

І	variables
~
VARIABLE_VALUEAdam/conv2d_174/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_174/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_175/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_175/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_176/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_176/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_114/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_114/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_115/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_115/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_174/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_174/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_175/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_175/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_176/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_176/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_114/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_114/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_115/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_115/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

 serving_default_conv2d_174_inputPlaceholder*/
_output_shapes
:џџџџџџџџџFF*
dtype0*$
shape:џџџџџџџџџFF
џ
StatefulPartitionedCallStatefulPartitionedCall serving_default_conv2d_174_inputconv2d_174/kernelconv2d_174/biasconv2d_175/kernelconv2d_175/biasconv2d_176/kernelconv2d_176/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_958523
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_174/kernel/Read/ReadVariableOp#conv2d_174/bias/Read/ReadVariableOp%conv2d_175/kernel/Read/ReadVariableOp#conv2d_175/bias/Read/ReadVariableOp%conv2d_176/kernel/Read/ReadVariableOp#conv2d_176/bias/Read/ReadVariableOp$dense_114/kernel/Read/ReadVariableOp"dense_114/bias/Read/ReadVariableOp$dense_115/kernel/Read/ReadVariableOp"dense_115/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_174/kernel/m/Read/ReadVariableOp*Adam/conv2d_174/bias/m/Read/ReadVariableOp,Adam/conv2d_175/kernel/m/Read/ReadVariableOp*Adam/conv2d_175/bias/m/Read/ReadVariableOp,Adam/conv2d_176/kernel/m/Read/ReadVariableOp*Adam/conv2d_176/bias/m/Read/ReadVariableOp+Adam/dense_114/kernel/m/Read/ReadVariableOp)Adam/dense_114/bias/m/Read/ReadVariableOp+Adam/dense_115/kernel/m/Read/ReadVariableOp)Adam/dense_115/bias/m/Read/ReadVariableOp,Adam/conv2d_174/kernel/v/Read/ReadVariableOp*Adam/conv2d_174/bias/v/Read/ReadVariableOp,Adam/conv2d_175/kernel/v/Read/ReadVariableOp*Adam/conv2d_175/bias/v/Read/ReadVariableOp,Adam/conv2d_176/kernel/v/Read/ReadVariableOp*Adam/conv2d_176/bias/v/Read/ReadVariableOp+Adam/dense_114/kernel/v/Read/ReadVariableOp)Adam/dense_114/bias/v/Read/ReadVariableOp+Adam/dense_115/kernel/v/Read/ReadVariableOp)Adam/dense_115/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_959056
О
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_174/kernelconv2d_174/biasconv2d_175/kernelconv2d_175/biasconv2d_176/kernelconv2d_176/biasdense_114/kerneldense_114/biasdense_115/kerneldense_115/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_174/kernel/mAdam/conv2d_174/bias/mAdam/conv2d_175/kernel/mAdam/conv2d_175/bias/mAdam/conv2d_176/kernel/mAdam/conv2d_176/bias/mAdam/dense_114/kernel/mAdam/dense_114/bias/mAdam/dense_115/kernel/mAdam/dense_115/bias/mAdam/conv2d_174/kernel/vAdam/conv2d_174/bias/vAdam/conv2d_175/kernel/vAdam/conv2d_175/bias/vAdam/conv2d_176/kernel/vAdam/conv2d_176/bias/vAdam/dense_114/kernel/vAdam/dense_114/bias/vAdam/dense_115/kernel/vAdam/dense_115/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_959183В
­
­
E__inference_dense_114_layer_call_and_return_conditional_losses_958860

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs

f
G__inference_dropout_137_layer_call_and_return_conditional_losses_958881

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeа
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
	

.__inference_sequential_59_layer_call_fn_958426
conv2d_174_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallconv2d_174_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_59_layer_call_and_return_conditional_losses_9584032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input
ъ
e
G__inference_dropout_135_layer_call_and_return_conditional_losses_958781

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
<
Ѓ
I__inference_sequential_59_layer_call_and_return_conditional_losses_958326
conv2d_174_input
conv2d_174_958075
conv2d_174_958077
conv2d_175_958133
conv2d_175_958135
conv2d_176_958191
conv2d_176_958193
dense_114_958263
dense_114_958265
dense_115_958320
dense_115_958322
identityЂ"conv2d_174/StatefulPartitionedCallЂ"conv2d_175/StatefulPartitionedCallЂ"conv2d_176/StatefulPartitionedCallЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallЂ#dropout_134/StatefulPartitionedCallЂ#dropout_135/StatefulPartitionedCallЂ#dropout_136/StatefulPartitionedCallЂ#dropout_137/StatefulPartitionedCallА
"conv2d_174/StatefulPartitionedCallStatefulPartitionedCallconv2d_174_inputconv2d_174_958075conv2d_174_958077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџDD *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_174_layer_call_and_return_conditional_losses_9580642$
"conv2d_174/StatefulPartitionedCall
!max_pooling2d_174/PartitionedCallPartitionedCall+conv2d_174/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_9580192#
!max_pooling2d_174/PartitionedCallЁ
#dropout_134/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_174/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580932%
#dropout_134/StatefulPartitionedCallЬ
"conv2d_175/StatefulPartitionedCallStatefulPartitionedCall,dropout_134/StatefulPartitionedCall:output:0conv2d_175_958133conv2d_175_958135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_175_layer_call_and_return_conditional_losses_9581222$
"conv2d_175/StatefulPartitionedCall
!max_pooling2d_175/PartitionedCallPartitionedCall+conv2d_175/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_9580312#
!max_pooling2d_175/PartitionedCallЧ
#dropout_135/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_175/PartitionedCall:output:0$^dropout_134/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581512%
#dropout_135/StatefulPartitionedCallЬ
"conv2d_176/StatefulPartitionedCallStatefulPartitionedCall,dropout_135/StatefulPartitionedCall:output:0conv2d_176_958191conv2d_176_958193*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_176_layer_call_and_return_conditional_losses_9581802$
"conv2d_176/StatefulPartitionedCall
!max_pooling2d_176/PartitionedCallPartitionedCall+conv2d_176/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_9580432#
!max_pooling2d_176/PartitionedCallЧ
#dropout_136/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_176/PartitionedCall:output:0$^dropout_135/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582092%
#dropout_136/StatefulPartitionedCall
flatten_58/PartitionedCallPartitionedCall,dropout_136/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_58_layer_call_and_return_conditional_losses_9582332
flatten_58/PartitionedCallЖ
!dense_114/StatefulPartitionedCallStatefulPartitionedCall#flatten_58/PartitionedCall:output:0dense_114_958263dense_114_958265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_114_layer_call_and_return_conditional_losses_9582522#
!dense_114/StatefulPartitionedCallП
#dropout_137/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0$^dropout_136/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582802%
#dropout_137/StatefulPartitionedCallП
!dense_115/StatefulPartitionedCallStatefulPartitionedCall,dropout_137/StatefulPartitionedCall:output:0dense_115_958320dense_115_958322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_115_layer_call_and_return_conditional_losses_9583092#
!dense_115/StatefulPartitionedCallЭ
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0#^conv2d_174/StatefulPartitionedCall#^conv2d_175/StatefulPartitionedCall#^conv2d_176/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall$^dropout_134/StatefulPartitionedCall$^dropout_135/StatefulPartitionedCall$^dropout_136/StatefulPartitionedCall$^dropout_137/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::2H
"conv2d_174/StatefulPartitionedCall"conv2d_174/StatefulPartitionedCall2H
"conv2d_175/StatefulPartitionedCall"conv2d_175/StatefulPartitionedCall2H
"conv2d_176/StatefulPartitionedCall"conv2d_176/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall2J
#dropout_134/StatefulPartitionedCall#dropout_134/StatefulPartitionedCall2J
#dropout_135/StatefulPartitionedCall#dropout_135/StatefulPartitionedCall2J
#dropout_136/StatefulPartitionedCall#dropout_136/StatefulPartitionedCall2J
#dropout_137/StatefulPartitionedCall#dropout_137/StatefulPartitionedCall:a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input
Ф
e
,__inference_dropout_135_layer_call_fn_958786

inputs
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_134_layer_call_and_return_conditional_losses_958098

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ"" :W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
МA
­
!__inference__wrapped_model_958013
conv2d_174_input;
7sequential_59_conv2d_174_conv2d_readvariableop_resource<
8sequential_59_conv2d_174_biasadd_readvariableop_resource;
7sequential_59_conv2d_175_conv2d_readvariableop_resource<
8sequential_59_conv2d_175_biasadd_readvariableop_resource;
7sequential_59_conv2d_176_conv2d_readvariableop_resource<
8sequential_59_conv2d_176_biasadd_readvariableop_resource:
6sequential_59_dense_114_matmul_readvariableop_resource;
7sequential_59_dense_114_biasadd_readvariableop_resource:
6sequential_59_dense_115_matmul_readvariableop_resource;
7sequential_59_dense_115_biasadd_readvariableop_resource
identityр
.sequential_59/conv2d_174/Conv2D/ReadVariableOpReadVariableOp7sequential_59_conv2d_174_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential_59/conv2d_174/Conv2D/ReadVariableOpљ
sequential_59/conv2d_174/Conv2DConv2Dconv2d_174_input6sequential_59/conv2d_174/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD *
paddingVALID*
strides
2!
sequential_59/conv2d_174/Conv2Dз
/sequential_59/conv2d_174/BiasAdd/ReadVariableOpReadVariableOp8sequential_59_conv2d_174_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_59/conv2d_174/BiasAdd/ReadVariableOpь
 sequential_59/conv2d_174/BiasAddBiasAdd(sequential_59/conv2d_174/Conv2D:output:07sequential_59/conv2d_174/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2"
 sequential_59/conv2d_174/BiasAddЋ
sequential_59/conv2d_174/TanhTanh)sequential_59/conv2d_174/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
sequential_59/conv2d_174/Tanhэ
'sequential_59/max_pooling2d_174/MaxPoolMaxPool!sequential_59/conv2d_174/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ"" *
ksize
*
paddingVALID*
strides
2)
'sequential_59/max_pooling2d_174/MaxPoolР
"sequential_59/dropout_134/IdentityIdentity0sequential_59/max_pooling2d_174/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2$
"sequential_59/dropout_134/Identityр
.sequential_59/conv2d_175/Conv2D/ReadVariableOpReadVariableOp7sequential_59_conv2d_175_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype020
.sequential_59/conv2d_175/Conv2D/ReadVariableOp
sequential_59/conv2d_175/Conv2DConv2D+sequential_59/dropout_134/Identity:output:06sequential_59/conv2d_175/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   *
paddingVALID*
strides
2!
sequential_59/conv2d_175/Conv2Dз
/sequential_59/conv2d_175/BiasAdd/ReadVariableOpReadVariableOp8sequential_59_conv2d_175_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_59/conv2d_175/BiasAdd/ReadVariableOpь
 sequential_59/conv2d_175/BiasAddBiasAdd(sequential_59/conv2d_175/Conv2D:output:07sequential_59/conv2d_175/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   2"
 sequential_59/conv2d_175/BiasAddЋ
sequential_59/conv2d_175/TanhTanh)sequential_59/conv2d_175/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
sequential_59/conv2d_175/Tanhэ
'sequential_59/max_pooling2d_175/MaxPoolMaxPool!sequential_59/conv2d_175/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2)
'sequential_59/max_pooling2d_175/MaxPoolР
"sequential_59/dropout_135/IdentityIdentity0sequential_59/max_pooling2d_175/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2$
"sequential_59/dropout_135/Identityр
.sequential_59/conv2d_176/Conv2D/ReadVariableOpReadVariableOp7sequential_59_conv2d_176_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.sequential_59/conv2d_176/Conv2D/ReadVariableOp
sequential_59/conv2d_176/Conv2DConv2D+sequential_59/dropout_135/Identity:output:06sequential_59/conv2d_176/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2!
sequential_59/conv2d_176/Conv2Dз
/sequential_59/conv2d_176/BiasAdd/ReadVariableOpReadVariableOp8sequential_59_conv2d_176_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_59/conv2d_176/BiasAdd/ReadVariableOpь
 sequential_59/conv2d_176/BiasAddBiasAdd(sequential_59/conv2d_176/Conv2D:output:07sequential_59/conv2d_176/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2"
 sequential_59/conv2d_176/BiasAddЋ
sequential_59/conv2d_176/SeluSelu)sequential_59/conv2d_176/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential_59/conv2d_176/Seluї
'sequential_59/max_pooling2d_176/MaxPoolMaxPool+sequential_59/conv2d_176/Selu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2)
'sequential_59/max_pooling2d_176/MaxPoolР
"sequential_59/dropout_136/IdentityIdentity0sequential_59/max_pooling2d_176/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2$
"sequential_59/dropout_136/Identity
sequential_59/flatten_58/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2 
sequential_59/flatten_58/Constи
 sequential_59/flatten_58/ReshapeReshape+sequential_59/dropout_136/Identity:output:0'sequential_59/flatten_58/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2"
 sequential_59/flatten_58/Reshapeж
-sequential_59/dense_114/MatMul/ReadVariableOpReadVariableOp6sequential_59_dense_114_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-sequential_59/dense_114/MatMul/ReadVariableOpо
sequential_59/dense_114/MatMulMatMul)sequential_59/flatten_58/Reshape:output:05sequential_59/dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
sequential_59/dense_114/MatMulд
.sequential_59/dense_114/BiasAdd/ReadVariableOpReadVariableOp7sequential_59_dense_114_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_59/dense_114/BiasAdd/ReadVariableOpс
sequential_59/dense_114/BiasAddBiasAdd(sequential_59/dense_114/MatMul:product:06sequential_59/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
sequential_59/dense_114/BiasAdd 
sequential_59/dense_114/ReluRelu(sequential_59/dense_114/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential_59/dense_114/ReluВ
"sequential_59/dropout_137/IdentityIdentity*sequential_59/dense_114/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2$
"sequential_59/dropout_137/Identityе
-sequential_59/dense_115/MatMul/ReadVariableOpReadVariableOp6sequential_59_dense_115_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02/
-sequential_59/dense_115/MatMul/ReadVariableOpр
sequential_59/dense_115/MatMulMatMul+sequential_59/dropout_137/Identity:output:05sequential_59/dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_59/dense_115/MatMulд
.sequential_59/dense_115/BiasAdd/ReadVariableOpReadVariableOp7sequential_59_dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_59/dense_115/BiasAdd/ReadVariableOpс
sequential_59/dense_115/BiasAddBiasAdd(sequential_59/dense_115/MatMul:product:06sequential_59/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_59/dense_115/BiasAddЉ
sequential_59/dense_115/SigmoidSigmoid(sequential_59/dense_115/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
sequential_59/dense_115/Sigmoidw
IdentityIdentity#sequential_59/dense_115/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF:::::::::::a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input

i
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_958043

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_136_layer_call_and_return_conditional_losses_958214

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_176_layer_call_and_return_conditional_losses_958180

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Selun
IdentityIdentitySelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ :::W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
с
f
G__inference_dropout_135_layer_call_and_return_conditional_losses_958151

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
с
f
G__inference_dropout_135_layer_call_and_return_conditional_losses_958776

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
с
f
G__inference_dropout_134_layer_call_and_return_conditional_losses_958093

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" *
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ"" :W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
с
f
G__inference_dropout_136_layer_call_and_return_conditional_losses_958209

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
H
,__inference_dropout_135_layer_call_fn_958791

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581562
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ї
ћ
.__inference_sequential_59_layer_call_fn_958672

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_59_layer_call_and_return_conditional_losses_9584032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
у
ћ
$__inference_signature_wrapper_958523
conv2d_174_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallconv2d_174_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_9580132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input
Б
N
2__inference_max_pooling2d_175_layer_call_fn_958037

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_9580312
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_175_layer_call_and_return_conditional_losses_958755

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ"" :::W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_174_layer_call_and_return_conditional_losses_958064

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџFF:::W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
р

*__inference_dense_114_layer_call_fn_958869

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_114_layer_call_and_return_conditional_losses_9582522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_958031

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ъ
e
G__inference_dropout_137_layer_call_and_return_conditional_losses_958285

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
б\
П
I__inference_sequential_59_layer_call_and_return_conditional_losses_958599

inputs-
)conv2d_174_conv2d_readvariableop_resource.
*conv2d_174_biasadd_readvariableop_resource-
)conv2d_175_conv2d_readvariableop_resource.
*conv2d_175_biasadd_readvariableop_resource-
)conv2d_176_conv2d_readvariableop_resource.
*conv2d_176_biasadd_readvariableop_resource,
(dense_114_matmul_readvariableop_resource-
)dense_114_biasadd_readvariableop_resource,
(dense_115_matmul_readvariableop_resource-
)dense_115_biasadd_readvariableop_resource
identityЖ
 conv2d_174/Conv2D/ReadVariableOpReadVariableOp)conv2d_174_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_174/Conv2D/ReadVariableOpХ
conv2d_174/Conv2DConv2Dinputs(conv2d_174/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD *
paddingVALID*
strides
2
conv2d_174/Conv2D­
!conv2d_174/BiasAdd/ReadVariableOpReadVariableOp*conv2d_174_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_174/BiasAdd/ReadVariableOpД
conv2d_174/BiasAddBiasAddconv2d_174/Conv2D:output:0)conv2d_174/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
conv2d_174/BiasAdd
conv2d_174/TanhTanhconv2d_174/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
conv2d_174/TanhУ
max_pooling2d_174/MaxPoolMaxPoolconv2d_174/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ"" *
ksize
*
paddingVALID*
strides
2
max_pooling2d_174/MaxPool{
dropout_134/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_134/dropout/ConstЛ
dropout_134/dropout/MulMul"max_pooling2d_174/MaxPool:output:0"dropout_134/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout_134/dropout/Mul
dropout_134/dropout/ShapeShape"max_pooling2d_174/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_134/dropout/Shapeќ
0dropout_134/dropout/random_uniform/RandomUniformRandomUniform"dropout_134/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" *
dtype0*
seedБџх)*
seed222
0dropout_134/dropout/random_uniform/RandomUniform
"dropout_134/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"dropout_134/dropout/GreaterEqual/yі
 dropout_134/dropout/GreaterEqualGreaterEqual9dropout_134/dropout/random_uniform/RandomUniform:output:0+dropout_134/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2"
 dropout_134/dropout/GreaterEqualЋ
dropout_134/dropout/CastCast$dropout_134/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ"" 2
dropout_134/dropout/CastВ
dropout_134/dropout/Mul_1Muldropout_134/dropout/Mul:z:0dropout_134/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout_134/dropout/Mul_1Ж
 conv2d_175/Conv2D/ReadVariableOpReadVariableOp)conv2d_175_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02"
 conv2d_175/Conv2D/ReadVariableOpм
conv2d_175/Conv2DConv2Ddropout_134/dropout/Mul_1:z:0(conv2d_175/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   *
paddingVALID*
strides
2
conv2d_175/Conv2D­
!conv2d_175/BiasAdd/ReadVariableOpReadVariableOp*conv2d_175_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_175/BiasAdd/ReadVariableOpД
conv2d_175/BiasAddBiasAddconv2d_175/Conv2D:output:0)conv2d_175/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
conv2d_175/BiasAdd
conv2d_175/TanhTanhconv2d_175/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
conv2d_175/TanhУ
max_pooling2d_175/MaxPoolMaxPoolconv2d_175/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_175/MaxPool{
dropout_135/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_135/dropout/ConstЛ
dropout_135/dropout/MulMul"max_pooling2d_175/MaxPool:output:0"dropout_135/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout_135/dropout/Mul
dropout_135/dropout/ShapeShape"max_pooling2d_175/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_135/dropout/Shapeќ
0dropout_135/dropout/random_uniform/RandomUniformRandomUniform"dropout_135/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed222
0dropout_135/dropout/random_uniform/RandomUniform
"dropout_135/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"dropout_135/dropout/GreaterEqual/yі
 dropout_135/dropout/GreaterEqualGreaterEqual9dropout_135/dropout/random_uniform/RandomUniform:output:0+dropout_135/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2"
 dropout_135/dropout/GreaterEqualЋ
dropout_135/dropout/CastCast$dropout_135/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ 2
dropout_135/dropout/CastВ
dropout_135/dropout/Mul_1Muldropout_135/dropout/Mul:z:0dropout_135/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout_135/dropout/Mul_1Ж
 conv2d_176/Conv2D/ReadVariableOpReadVariableOp)conv2d_176_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_176/Conv2D/ReadVariableOpм
conv2d_176/Conv2DConv2Ddropout_135/dropout/Mul_1:z:0(conv2d_176/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_176/Conv2D­
!conv2d_176/BiasAdd/ReadVariableOpReadVariableOp*conv2d_176_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_176/BiasAdd/ReadVariableOpД
conv2d_176/BiasAddBiasAddconv2d_176/Conv2D:output:0)conv2d_176/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_176/BiasAdd
conv2d_176/SeluSeluconv2d_176/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_176/SeluЭ
max_pooling2d_176/MaxPoolMaxPoolconv2d_176/Selu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_176/MaxPool{
dropout_136/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_136/dropout/ConstЛ
dropout_136/dropout/MulMul"max_pooling2d_176/MaxPool:output:0"dropout_136/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout_136/dropout/Mul
dropout_136/dropout/ShapeShape"max_pooling2d_176/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_136/dropout/Shapeќ
0dropout_136/dropout/random_uniform/RandomUniformRandomUniform"dropout_136/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed222
0dropout_136/dropout/random_uniform/RandomUniform
"dropout_136/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"dropout_136/dropout/GreaterEqual/yі
 dropout_136/dropout/GreaterEqualGreaterEqual9dropout_136/dropout/random_uniform/RandomUniform:output:0+dropout_136/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2"
 dropout_136/dropout/GreaterEqualЋ
dropout_136/dropout/CastCast$dropout_136/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout_136/dropout/CastВ
dropout_136/dropout/Mul_1Muldropout_136/dropout/Mul:z:0dropout_136/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout_136/dropout/Mul_1u
flatten_58/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten_58/Const 
flatten_58/ReshapeReshapedropout_136/dropout/Mul_1:z:0flatten_58/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten_58/ReshapeЌ
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02!
dense_114/MatMul/ReadVariableOpІ
dense_114/MatMulMatMulflatten_58/Reshape:output:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/MatMulЊ
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_114/BiasAdd/ReadVariableOpЉ
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/BiasAddv
dense_114/ReluReludense_114/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/Relu{
dropout_137/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_137/dropout/Const­
dropout_137/dropout/MulMuldense_114/Relu:activations:0"dropout_137/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_137/dropout/Mul
dropout_137/dropout/ShapeShapedense_114/Relu:activations:0*
T0*
_output_shapes
:2
dropout_137/dropout/Shapeє
0dropout_137/dropout/random_uniform/RandomUniformRandomUniform"dropout_137/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed222
0dropout_137/dropout/random_uniform/RandomUniform
"dropout_137/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"dropout_137/dropout/GreaterEqual/yю
 dropout_137/dropout/GreaterEqualGreaterEqual9dropout_137/dropout/random_uniform/RandomUniform:output:0+dropout_137/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2"
 dropout_137/dropout/GreaterEqualЃ
dropout_137/dropout/CastCast$dropout_137/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout_137/dropout/CastЊ
dropout_137/dropout/Mul_1Muldropout_137/dropout/Mul:z:0dropout_137/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_137/dropout/Mul_1Ћ
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_115/MatMul/ReadVariableOpЈ
dense_115/MatMulMatMuldropout_137/dropout/Mul_1:z:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/MatMulЊ
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_115/BiasAdd/ReadVariableOpЉ
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/BiasAdd
dense_115/SigmoidSigmoiddense_115/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/Sigmoidi
IdentityIdentitydense_115/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF:::::::::::W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
ЄЅ
Д
"__inference__traced_restore_959183
file_prefix&
"assignvariableop_conv2d_174_kernel&
"assignvariableop_1_conv2d_174_bias(
$assignvariableop_2_conv2d_175_kernel&
"assignvariableop_3_conv2d_175_bias(
$assignvariableop_4_conv2d_176_kernel&
"assignvariableop_5_conv2d_176_bias'
#assignvariableop_6_dense_114_kernel%
!assignvariableop_7_dense_114_bias'
#assignvariableop_8_dense_115_kernel%
!assignvariableop_9_dense_115_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_10
,assignvariableop_19_adam_conv2d_174_kernel_m.
*assignvariableop_20_adam_conv2d_174_bias_m0
,assignvariableop_21_adam_conv2d_175_kernel_m.
*assignvariableop_22_adam_conv2d_175_bias_m0
,assignvariableop_23_adam_conv2d_176_kernel_m.
*assignvariableop_24_adam_conv2d_176_bias_m/
+assignvariableop_25_adam_dense_114_kernel_m-
)assignvariableop_26_adam_dense_114_bias_m/
+assignvariableop_27_adam_dense_115_kernel_m-
)assignvariableop_28_adam_dense_115_bias_m0
,assignvariableop_29_adam_conv2d_174_kernel_v.
*assignvariableop_30_adam_conv2d_174_bias_v0
,assignvariableop_31_adam_conv2d_175_kernel_v.
*assignvariableop_32_adam_conv2d_175_bias_v0
,assignvariableop_33_adam_conv2d_176_kernel_v.
*assignvariableop_34_adam_conv2d_176_bias_v/
+assignvariableop_35_adam_dense_114_kernel_v-
)assignvariableop_36_adam_dense_114_bias_v/
+assignvariableop_37_adam_dense_115_kernel_v-
)assignvariableop_38_adam_dense_115_bias_v
identity_40ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesі
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesЃ
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЁ
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_174_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ї
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_174_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Љ
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_175_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ї
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_175_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Љ
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_176_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ї
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_176_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ј
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_114_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_114_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ј
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_115_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9І
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_115_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ї
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ў
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ё
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ё
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ѓ
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Д
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_174_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20В
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_174_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_conv2d_175_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22В
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_conv2d_175_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Д
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_conv2d_176_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24В
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv2d_176_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Г
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_114_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Б
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_114_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Г
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_115_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_115_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Д
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_conv2d_174_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30В
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_conv2d_174_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Д
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_175_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32В
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_175_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Д
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_176_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34В
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_176_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Г
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_114_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Б
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_114_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Г
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_115_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Б
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_115_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39Ћ
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*Г
_input_shapesЁ
: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
О
b
F__inference_flatten_58_layer_call_and_return_conditional_losses_958844

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї
ћ
.__inference_sequential_59_layer_call_fn_958697

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_59_layer_call_and_return_conditional_losses_9584652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
­
­
E__inference_dense_114_layer_call_and_return_conditional_losses_958252

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Ѕ5

I__inference_sequential_59_layer_call_and_return_conditional_losses_958465

inputs
conv2d_174_958431
conv2d_174_958433
conv2d_175_958438
conv2d_175_958440
conv2d_176_958445
conv2d_176_958447
dense_114_958453
dense_114_958455
dense_115_958459
dense_115_958461
identityЂ"conv2d_174/StatefulPartitionedCallЂ"conv2d_175/StatefulPartitionedCallЂ"conv2d_176/StatefulPartitionedCallЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallІ
"conv2d_174/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_174_958431conv2d_174_958433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџDD *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_174_layer_call_and_return_conditional_losses_9580642$
"conv2d_174/StatefulPartitionedCall
!max_pooling2d_174/PartitionedCallPartitionedCall+conv2d_174/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_9580192#
!max_pooling2d_174/PartitionedCall
dropout_134/PartitionedCallPartitionedCall*max_pooling2d_174/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580982
dropout_134/PartitionedCallФ
"conv2d_175/StatefulPartitionedCallStatefulPartitionedCall$dropout_134/PartitionedCall:output:0conv2d_175_958438conv2d_175_958440*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_175_layer_call_and_return_conditional_losses_9581222$
"conv2d_175/StatefulPartitionedCall
!max_pooling2d_175/PartitionedCallPartitionedCall+conv2d_175/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_9580312#
!max_pooling2d_175/PartitionedCall
dropout_135/PartitionedCallPartitionedCall*max_pooling2d_175/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581562
dropout_135/PartitionedCallФ
"conv2d_176/StatefulPartitionedCallStatefulPartitionedCall$dropout_135/PartitionedCall:output:0conv2d_176_958445conv2d_176_958447*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_176_layer_call_and_return_conditional_losses_9581802$
"conv2d_176/StatefulPartitionedCall
!max_pooling2d_176/PartitionedCallPartitionedCall+conv2d_176/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_9580432#
!max_pooling2d_176/PartitionedCall
dropout_136/PartitionedCallPartitionedCall*max_pooling2d_176/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582142
dropout_136/PartitionedCallљ
flatten_58/PartitionedCallPartitionedCall$dropout_136/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_58_layer_call_and_return_conditional_losses_9582332
flatten_58/PartitionedCallЖ
!dense_114/StatefulPartitionedCallStatefulPartitionedCall#flatten_58/PartitionedCall:output:0dense_114_958453dense_114_958455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_114_layer_call_and_return_conditional_losses_9582522#
!dense_114/StatefulPartitionedCall
dropout_137/PartitionedCallPartitionedCall*dense_114/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582852
dropout_137/PartitionedCallЗ
!dense_115/StatefulPartitionedCallStatefulPartitionedCall$dropout_137/PartitionedCall:output:0dense_115_958459dense_115_958461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_115_layer_call_and_return_conditional_losses_9583092#
!dense_115/StatefulPartitionedCallЕ
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0#^conv2d_174/StatefulPartitionedCall#^conv2d_175/StatefulPartitionedCall#^conv2d_176/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::2H
"conv2d_174/StatefulPartitionedCall"conv2d_174/StatefulPartitionedCall2H
"conv2d_175/StatefulPartitionedCall"conv2d_175/StatefulPartitionedCall2H
"conv2d_176/StatefulPartitionedCall"conv2d_176/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_134_layer_call_and_return_conditional_losses_958734

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ"" :W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs

f
G__inference_dropout_137_layer_call_and_return_conditional_losses_958280

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeа
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_174_layer_call_and_return_conditional_losses_958708

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџFF:::W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
Ъ
e
G__inference_dropout_137_layer_call_and_return_conditional_losses_958886

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


+__inference_conv2d_175_layer_call_fn_958764

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_175_layer_call_and_return_conditional_losses_9581222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ"" ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
Ј
G
+__inference_flatten_58_layer_call_fn_958849

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_58_layer_call_and_return_conditional_losses_9582332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
­
E__inference_dense_115_layer_call_and_return_conditional_losses_958907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
b
F__inference_flatten_58_layer_call_and_return_conditional_losses_958233

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Є
e
,__inference_dropout_137_layer_call_fn_958891

inputs
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_135_layer_call_and_return_conditional_losses_958156

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
И
H
,__inference_dropout_136_layer_call_fn_958838

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582142
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
4
П
I__inference_sequential_59_layer_call_and_return_conditional_losses_958647

inputs-
)conv2d_174_conv2d_readvariableop_resource.
*conv2d_174_biasadd_readvariableop_resource-
)conv2d_175_conv2d_readvariableop_resource.
*conv2d_175_biasadd_readvariableop_resource-
)conv2d_176_conv2d_readvariableop_resource.
*conv2d_176_biasadd_readvariableop_resource,
(dense_114_matmul_readvariableop_resource-
)dense_114_biasadd_readvariableop_resource,
(dense_115_matmul_readvariableop_resource-
)dense_115_biasadd_readvariableop_resource
identityЖ
 conv2d_174/Conv2D/ReadVariableOpReadVariableOp)conv2d_174_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_174/Conv2D/ReadVariableOpХ
conv2d_174/Conv2DConv2Dinputs(conv2d_174/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD *
paddingVALID*
strides
2
conv2d_174/Conv2D­
!conv2d_174/BiasAdd/ReadVariableOpReadVariableOp*conv2d_174_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_174/BiasAdd/ReadVariableOpД
conv2d_174/BiasAddBiasAddconv2d_174/Conv2D:output:0)conv2d_174/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
conv2d_174/BiasAdd
conv2d_174/TanhTanhconv2d_174/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџDD 2
conv2d_174/TanhУ
max_pooling2d_174/MaxPoolMaxPoolconv2d_174/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ"" *
ksize
*
paddingVALID*
strides
2
max_pooling2d_174/MaxPool
dropout_134/IdentityIdentity"max_pooling2d_174/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout_134/IdentityЖ
 conv2d_175/Conv2D/ReadVariableOpReadVariableOp)conv2d_175_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02"
 conv2d_175/Conv2D/ReadVariableOpм
conv2d_175/Conv2DConv2Ddropout_134/Identity:output:0(conv2d_175/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   *
paddingVALID*
strides
2
conv2d_175/Conv2D­
!conv2d_175/BiasAdd/ReadVariableOpReadVariableOp*conv2d_175_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_175/BiasAdd/ReadVariableOpД
conv2d_175/BiasAddBiasAddconv2d_175/Conv2D:output:0)conv2d_175/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
conv2d_175/BiasAdd
conv2d_175/TanhTanhconv2d_175/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
conv2d_175/TanhУ
max_pooling2d_175/MaxPoolMaxPoolconv2d_175/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_175/MaxPool
dropout_135/IdentityIdentity"max_pooling2d_175/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
dropout_135/IdentityЖ
 conv2d_176/Conv2D/ReadVariableOpReadVariableOp)conv2d_176_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_176/Conv2D/ReadVariableOpм
conv2d_176/Conv2DConv2Ddropout_135/Identity:output:0(conv2d_176/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_176/Conv2D­
!conv2d_176/BiasAdd/ReadVariableOpReadVariableOp*conv2d_176_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_176/BiasAdd/ReadVariableOpД
conv2d_176/BiasAddBiasAddconv2d_176/Conv2D:output:0)conv2d_176/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_176/BiasAdd
conv2d_176/SeluSeluconv2d_176/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_176/SeluЭ
max_pooling2d_176/MaxPoolMaxPoolconv2d_176/Selu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_176/MaxPool
dropout_136/IdentityIdentity"max_pooling2d_176/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout_136/Identityu
flatten_58/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten_58/Const 
flatten_58/ReshapeReshapedropout_136/Identity:output:0flatten_58/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten_58/ReshapeЌ
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02!
dense_114/MatMul/ReadVariableOpІ
dense_114/MatMulMatMulflatten_58/Reshape:output:0'dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/MatMulЊ
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_114/BiasAdd/ReadVariableOpЉ
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/BiasAddv
dense_114/ReluReludense_114/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_114/Relu
dropout_137/IdentityIdentitydense_114/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_137/IdentityЋ
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_115/MatMul/ReadVariableOpЈ
dense_115/MatMulMatMuldropout_137/Identity:output:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/MatMulЊ
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_115/BiasAdd/ReadVariableOpЉ
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/BiasAdd
dense_115/SigmoidSigmoiddense_115/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_115/Sigmoidi
IdentityIdentitydense_115/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF:::::::::::W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs

H
,__inference_dropout_137_layer_call_fn_958896

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582852
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
T
І
__inference__traced_save_959056
file_prefix0
,savev2_conv2d_174_kernel_read_readvariableop.
*savev2_conv2d_174_bias_read_readvariableop0
,savev2_conv2d_175_kernel_read_readvariableop.
*savev2_conv2d_175_bias_read_readvariableop0
,savev2_conv2d_176_kernel_read_readvariableop.
*savev2_conv2d_176_bias_read_readvariableop/
+savev2_dense_114_kernel_read_readvariableop-
)savev2_dense_114_bias_read_readvariableop/
+savev2_dense_115_kernel_read_readvariableop-
)savev2_dense_115_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_174_kernel_m_read_readvariableop5
1savev2_adam_conv2d_174_bias_m_read_readvariableop7
3savev2_adam_conv2d_175_kernel_m_read_readvariableop5
1savev2_adam_conv2d_175_bias_m_read_readvariableop7
3savev2_adam_conv2d_176_kernel_m_read_readvariableop5
1savev2_adam_conv2d_176_bias_m_read_readvariableop6
2savev2_adam_dense_114_kernel_m_read_readvariableop4
0savev2_adam_dense_114_bias_m_read_readvariableop6
2savev2_adam_dense_115_kernel_m_read_readvariableop4
0savev2_adam_dense_115_bias_m_read_readvariableop7
3savev2_adam_conv2d_174_kernel_v_read_readvariableop5
1savev2_adam_conv2d_174_bias_v_read_readvariableop7
3savev2_adam_conv2d_175_kernel_v_read_readvariableop5
1savev2_adam_conv2d_175_bias_v_read_readvariableop7
3savev2_adam_conv2d_176_kernel_v_read_readvariableop5
1savev2_adam_conv2d_176_bias_v_read_readvariableop6
2savev2_adam_dense_114_kernel_v_read_readvariableop4
0savev2_adam_dense_114_bias_v_read_readvariableop6
2savev2_adam_dense_115_kernel_v_read_readvariableop4
0savev2_adam_dense_115_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ccee1e9b7ee840769a43b0500f5e1717/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesи
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesџ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_174_kernel_read_readvariableop*savev2_conv2d_174_bias_read_readvariableop,savev2_conv2d_175_kernel_read_readvariableop*savev2_conv2d_175_bias_read_readvariableop,savev2_conv2d_176_kernel_read_readvariableop*savev2_conv2d_176_bias_read_readvariableop+savev2_dense_114_kernel_read_readvariableop)savev2_dense_114_bias_read_readvariableop+savev2_dense_115_kernel_read_readvariableop)savev2_dense_115_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_174_kernel_m_read_readvariableop1savev2_adam_conv2d_174_bias_m_read_readvariableop3savev2_adam_conv2d_175_kernel_m_read_readvariableop1savev2_adam_conv2d_175_bias_m_read_readvariableop3savev2_adam_conv2d_176_kernel_m_read_readvariableop1savev2_adam_conv2d_176_bias_m_read_readvariableop2savev2_adam_dense_114_kernel_m_read_readvariableop0savev2_adam_dense_114_bias_m_read_readvariableop2savev2_adam_dense_115_kernel_m_read_readvariableop0savev2_adam_dense_115_bias_m_read_readvariableop3savev2_adam_conv2d_174_kernel_v_read_readvariableop1savev2_adam_conv2d_174_bias_v_read_readvariableop3savev2_adam_conv2d_175_kernel_v_read_readvariableop1savev2_adam_conv2d_175_bias_v_read_readvariableop3savev2_adam_conv2d_176_kernel_v_read_readvariableop1savev2_adam_conv2d_176_bias_v_read_readvariableop2savev2_adam_dense_114_kernel_v_read_readvariableop0savev2_adam_dense_114_bias_v_read_readvariableop2savev2_adam_dense_115_kernel_v_read_readvariableop0savev2_adam_dense_115_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ц
_input_shapesд
б: : : :  : : @:@:	Р@:@:@:: : : : : : : : : : : :  : : @:@:	Р@:@:@:: : :  : : @:@:	Р@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	Р@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	Р@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:%$!

_output_shapes
:	Р@: %

_output_shapes
:@:$& 

_output_shapes

:@: '

_output_shapes
::(

_output_shapes
: 
	

.__inference_sequential_59_layer_call_fn_958488
conv2d_174_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallconv2d_174_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_59_layer_call_and_return_conditional_losses_9584652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input


+__inference_conv2d_174_layer_call_fn_958717

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџDD *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_174_layer_call_and_return_conditional_losses_9580642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџDD 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџFF::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
с
f
G__inference_dropout_134_layer_call_and_return_conditional_losses_958729

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" *
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ"" :W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
Б
N
2__inference_max_pooling2d_174_layer_call_fn_958025

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_9580192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_958019

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У5

I__inference_sequential_59_layer_call_and_return_conditional_losses_958363
conv2d_174_input
conv2d_174_958329
conv2d_174_958331
conv2d_175_958336
conv2d_175_958338
conv2d_176_958343
conv2d_176_958345
dense_114_958351
dense_114_958353
dense_115_958357
dense_115_958359
identityЂ"conv2d_174/StatefulPartitionedCallЂ"conv2d_175/StatefulPartitionedCallЂ"conv2d_176/StatefulPartitionedCallЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallА
"conv2d_174/StatefulPartitionedCallStatefulPartitionedCallconv2d_174_inputconv2d_174_958329conv2d_174_958331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџDD *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_174_layer_call_and_return_conditional_losses_9580642$
"conv2d_174/StatefulPartitionedCall
!max_pooling2d_174/PartitionedCallPartitionedCall+conv2d_174/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_9580192#
!max_pooling2d_174/PartitionedCall
dropout_134/PartitionedCallPartitionedCall*max_pooling2d_174/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580982
dropout_134/PartitionedCallФ
"conv2d_175/StatefulPartitionedCallStatefulPartitionedCall$dropout_134/PartitionedCall:output:0conv2d_175_958336conv2d_175_958338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_175_layer_call_and_return_conditional_losses_9581222$
"conv2d_175/StatefulPartitionedCall
!max_pooling2d_175/PartitionedCallPartitionedCall+conv2d_175/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_9580312#
!max_pooling2d_175/PartitionedCall
dropout_135/PartitionedCallPartitionedCall*max_pooling2d_175/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581562
dropout_135/PartitionedCallФ
"conv2d_176/StatefulPartitionedCallStatefulPartitionedCall$dropout_135/PartitionedCall:output:0conv2d_176_958343conv2d_176_958345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_176_layer_call_and_return_conditional_losses_9581802$
"conv2d_176/StatefulPartitionedCall
!max_pooling2d_176/PartitionedCallPartitionedCall+conv2d_176/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_9580432#
!max_pooling2d_176/PartitionedCall
dropout_136/PartitionedCallPartitionedCall*max_pooling2d_176/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582142
dropout_136/PartitionedCallљ
flatten_58/PartitionedCallPartitionedCall$dropout_136/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_58_layer_call_and_return_conditional_losses_9582332
flatten_58/PartitionedCallЖ
!dense_114/StatefulPartitionedCallStatefulPartitionedCall#flatten_58/PartitionedCall:output:0dense_114_958351dense_114_958353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_114_layer_call_and_return_conditional_losses_9582522#
!dense_114/StatefulPartitionedCall
dropout_137/PartitionedCallPartitionedCall*dense_114/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582852
dropout_137/PartitionedCallЗ
!dense_115/StatefulPartitionedCallStatefulPartitionedCall$dropout_137/PartitionedCall:output:0dense_115_958357dense_115_958359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_115_layer_call_and_return_conditional_losses_9583092#
!dense_115/StatefulPartitionedCallЕ
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0#^conv2d_174/StatefulPartitionedCall#^conv2d_175/StatefulPartitionedCall#^conv2d_176/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::2H
"conv2d_174/StatefulPartitionedCall"conv2d_174/StatefulPartitionedCall2H
"conv2d_175/StatefulPartitionedCall"conv2d_175/StatefulPartitionedCall2H
"conv2d_176/StatefulPartitionedCall"conv2d_176/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall:a ]
/
_output_shapes
:џџџџџџџџџFF
*
_user_specified_nameconv2d_174_input
Ф
e
,__inference_dropout_136_layer_call_fn_958833

inputs
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ъ
e
G__inference_dropout_136_layer_call_and_return_conditional_losses_958828

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


+__inference_conv2d_176_layer_call_fn_958811

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_176_layer_call_and_return_conditional_losses_9581802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
с
f
G__inference_dropout_136_layer_call_and_return_conditional_losses_958823

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeи
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*
seedБџх)*
seed22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
H
,__inference_dropout_134_layer_call_fn_958744

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580982
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ"" :W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
о

*__inference_dense_115_layer_call_fn_958916

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_115_layer_call_and_return_conditional_losses_9583092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
­
E__inference_dense_115_layer_call_and_return_conditional_losses_958309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Б
N
2__inference_max_pooling2d_176_layer_call_fn_958049

inputs
identityю
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_9580432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї;

I__inference_sequential_59_layer_call_and_return_conditional_losses_958403

inputs
conv2d_174_958369
conv2d_174_958371
conv2d_175_958376
conv2d_175_958378
conv2d_176_958383
conv2d_176_958385
dense_114_958391
dense_114_958393
dense_115_958397
dense_115_958399
identityЂ"conv2d_174/StatefulPartitionedCallЂ"conv2d_175/StatefulPartitionedCallЂ"conv2d_176/StatefulPartitionedCallЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallЂ#dropout_134/StatefulPartitionedCallЂ#dropout_135/StatefulPartitionedCallЂ#dropout_136/StatefulPartitionedCallЂ#dropout_137/StatefulPartitionedCallІ
"conv2d_174/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_174_958369conv2d_174_958371*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџDD *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_174_layer_call_and_return_conditional_losses_9580642$
"conv2d_174/StatefulPartitionedCall
!max_pooling2d_174/PartitionedCallPartitionedCall+conv2d_174/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_9580192#
!max_pooling2d_174/PartitionedCallЁ
#dropout_134/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_174/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580932%
#dropout_134/StatefulPartitionedCallЬ
"conv2d_175/StatefulPartitionedCallStatefulPartitionedCall,dropout_134/StatefulPartitionedCall:output:0conv2d_175_958376conv2d_175_958378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_175_layer_call_and_return_conditional_losses_9581222$
"conv2d_175/StatefulPartitionedCall
!max_pooling2d_175/PartitionedCallPartitionedCall+conv2d_175/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_9580312#
!max_pooling2d_175/PartitionedCallЧ
#dropout_135/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_175/PartitionedCall:output:0$^dropout_134/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_135_layer_call_and_return_conditional_losses_9581512%
#dropout_135/StatefulPartitionedCallЬ
"conv2d_176/StatefulPartitionedCallStatefulPartitionedCall,dropout_135/StatefulPartitionedCall:output:0conv2d_176_958383conv2d_176_958385*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_176_layer_call_and_return_conditional_losses_9581802$
"conv2d_176/StatefulPartitionedCall
!max_pooling2d_176/PartitionedCallPartitionedCall+conv2d_176/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_9580432#
!max_pooling2d_176/PartitionedCallЧ
#dropout_136/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_176/PartitionedCall:output:0$^dropout_135/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_136_layer_call_and_return_conditional_losses_9582092%
#dropout_136/StatefulPartitionedCall
flatten_58/PartitionedCallPartitionedCall,dropout_136/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_58_layer_call_and_return_conditional_losses_9582332
flatten_58/PartitionedCallЖ
!dense_114/StatefulPartitionedCallStatefulPartitionedCall#flatten_58/PartitionedCall:output:0dense_114_958391dense_114_958393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_114_layer_call_and_return_conditional_losses_9582522#
!dense_114/StatefulPartitionedCallП
#dropout_137/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0$^dropout_136/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_137_layer_call_and_return_conditional_losses_9582802%
#dropout_137/StatefulPartitionedCallП
!dense_115/StatefulPartitionedCallStatefulPartitionedCall,dropout_137/StatefulPartitionedCall:output:0dense_115_958397dense_115_958399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_115_layer_call_and_return_conditional_losses_9583092#
!dense_115/StatefulPartitionedCallЭ
IdentityIdentity*dense_115/StatefulPartitionedCall:output:0#^conv2d_174/StatefulPartitionedCall#^conv2d_175/StatefulPartitionedCall#^conv2d_176/StatefulPartitionedCall"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall$^dropout_134/StatefulPartitionedCall$^dropout_135/StatefulPartitionedCall$^dropout_136/StatefulPartitionedCall$^dropout_137/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџFF::::::::::2H
"conv2d_174/StatefulPartitionedCall"conv2d_174/StatefulPartitionedCall2H
"conv2d_175/StatefulPartitionedCall"conv2d_175/StatefulPartitionedCall2H
"conv2d_176/StatefulPartitionedCall"conv2d_176/StatefulPartitionedCall2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall2J
#dropout_134/StatefulPartitionedCall#dropout_134/StatefulPartitionedCall2J
#dropout_135/StatefulPartitionedCall#dropout_135/StatefulPartitionedCall2J
#dropout_136/StatefulPartitionedCall#dropout_136/StatefulPartitionedCall2J
#dropout_137/StatefulPartitionedCall#dropout_137/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџFF
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_175_layer_call_and_return_conditional_losses_958122

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ   2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ   2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ"" :::W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs
	
Ў
F__inference_conv2d_176_layer_call_and_return_conditional_losses_958802

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
SeluSeluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Selun
IdentityIdentitySelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ :::W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ф
e
,__inference_dropout_134_layer_call_fn_958739

inputs
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_134_layer_call_and_return_conditional_losses_9580932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ"" 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ"" 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ"" 
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ц
serving_defaultВ
U
conv2d_174_inputA
"serving_default_conv2d_174_input:0џџџџџџџџџFF=
	dense_1150
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ф§
УV
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
М_default_save_signature
Н__call__
+О&call_and_return_all_conditional_losses"ЧR
_tf_keras_sequentialЈR{"class_name": "Sequential", "name": "sequential_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_59", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_174_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_174", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_174", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_134", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_175", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_175", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_135", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_176", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_176", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_136", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Flatten", "config": {"name": "flatten_58", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_137", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Dense", "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 70, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_59", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_174_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_174", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_174", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_134", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_175", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_175", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_135", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_176", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_176", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_136", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Flatten", "config": {"name": "flatten_58", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_137", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}, {"class_name": "Dense", "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.800000011920929, "beta_2": 0.8999999761581421, "epsilon": 1e-07, "amsgrad": false}}}}


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"ъ	
_tf_keras_layerа	{"class_name": "Conv2D", "name": "conv2d_174", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_174", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 70, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 70, 3]}}

	variables
regularization_losses
trainable_variables
	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"є
_tf_keras_layerк{"class_name": "MaxPooling2D", "name": "max_pooling2d_174", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_174", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
	variables
regularization_losses
 trainable_variables
!	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Dropout", "name": "dropout_134", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_134", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}



"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"ы
_tf_keras_layerб{"class_name": "Conv2D", "name": "conv2d_175", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_175", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 32]}}

(	variables
)regularization_losses
*trainable_variables
+	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"є
_tf_keras_layerк{"class_name": "MaxPooling2D", "name": "max_pooling2d_175", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_175", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
,	variables
-regularization_losses
.trainable_variables
/	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Dropout", "name": "dropout_135", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_135", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}
ё	

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"Ъ
_tf_keras_layerА{"class_name": "Conv2D", "name": "conv2d_176", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_176", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}

6	variables
7regularization_losses
8trainable_variables
9	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"є
_tf_keras_layerк{"class_name": "MaxPooling2D", "name": "max_pooling2d_176", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_176", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
:	variables
;regularization_losses
<trainable_variables
=	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Dropout", "name": "dropout_136", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_136", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}
ъ
>	variables
?regularization_losses
@trainable_variables
A	keras_api
б__call__
+в&call_and_return_all_conditional_losses"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_58", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ђ

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
г__call__
+д&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_114", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_114", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3136]}}
ш
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Dropout", "name": "dropout_137", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_137", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": 1}}


Lkernel
Mbias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
з__call__
+и&call_and_return_all_conditional_losses"ъ
_tf_keras_layerа{"class_name": "Dense", "name": "dense_115", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_115", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}

Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemЈmЉ"mЊ#mЋ0mЌ1m­BmЎCmЏLmАMmБvВvГ"vД#vЕ0vЖ1vЗBvИCvЙLvКMvЛ"
	optimizer
 "
trackable_list_wrapper
f
0
1
"2
#3
04
15
B6
C7
L8
M9"
trackable_list_wrapper
f
0
1
"2
#3
04
15
B6
C7
L8
M9"
trackable_list_wrapper
Ю
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
regularization_losses
Znon_trainable_variables

[layers
trainable_variables
	variables
Н__call__
М_default_save_signature
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
-
йserving_default"
signature_map
+:) 2conv2d_174/kernel
: 2conv2d_174/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
\metrics
	variables
]layer_metrics
^layer_regularization_losses
regularization_losses

_layers
trainable_variables
`non_trainable_variables
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
ametrics
	variables
blayer_metrics
clayer_regularization_losses
regularization_losses

dlayers
trainable_variables
enon_trainable_variables
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
fmetrics
	variables
glayer_metrics
hlayer_regularization_losses
regularization_losses

ilayers
 trainable_variables
jnon_trainable_variables
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
+:)  2conv2d_175/kernel
: 2conv2d_175/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
А
kmetrics
$	variables
llayer_metrics
mlayer_regularization_losses
%regularization_losses

nlayers
&trainable_variables
onon_trainable_variables
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
pmetrics
(	variables
qlayer_metrics
rlayer_regularization_losses
)regularization_losses

slayers
*trainable_variables
tnon_trainable_variables
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
umetrics
,	variables
vlayer_metrics
wlayer_regularization_losses
-regularization_losses

xlayers
.trainable_variables
ynon_trainable_variables
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
+:) @2conv2d_176/kernel
:@2conv2d_176/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
А
zmetrics
2	variables
{layer_metrics
|layer_regularization_losses
3regularization_losses

}layers
4trainable_variables
~non_trainable_variables
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Д
metrics
6	variables
layer_metrics
 layer_regularization_losses
7regularization_losses
layers
8trainable_variables
non_trainable_variables
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
:	variables
layer_metrics
 layer_regularization_losses
;regularization_losses
layers
<trainable_variables
non_trainable_variables
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
>	variables
layer_metrics
 layer_regularization_losses
?regularization_losses
layers
@trainable_variables
non_trainable_variables
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
#:!	Р@2dense_114/kernel
:@2dense_114/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
Е
metrics
D	variables
layer_metrics
 layer_regularization_losses
Eregularization_losses
layers
Ftrainable_variables
non_trainable_variables
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
metrics
H	variables
layer_metrics
 layer_regularization_losses
Iregularization_losses
layers
Jtrainable_variables
non_trainable_variables
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
": @2dense_115/kernel
:2dense_115/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
Е
metrics
N	variables
layer_metrics
 layer_regularization_losses
Oregularization_losses
layers
Ptrainable_variables
non_trainable_variables
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

total

 count
Ё	variables
Ђ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ

Ѓtotal

Єcount
Ѕ
_fn_kwargs
І	variables
Ї	keras_api"Г
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
.
І	variables"
_generic_user_object
0:. 2Adam/conv2d_174/kernel/m
":  2Adam/conv2d_174/bias/m
0:.  2Adam/conv2d_175/kernel/m
":  2Adam/conv2d_175/bias/m
0:. @2Adam/conv2d_176/kernel/m
": @2Adam/conv2d_176/bias/m
(:&	Р@2Adam/dense_114/kernel/m
!:@2Adam/dense_114/bias/m
':%@2Adam/dense_115/kernel/m
!:2Adam/dense_115/bias/m
0:. 2Adam/conv2d_174/kernel/v
":  2Adam/conv2d_174/bias/v
0:.  2Adam/conv2d_175/kernel/v
":  2Adam/conv2d_175/bias/v
0:. @2Adam/conv2d_176/kernel/v
": @2Adam/conv2d_176/bias/v
(:&	Р@2Adam/dense_114/kernel/v
!:@2Adam/dense_114/bias/v
':%@2Adam/dense_115/kernel/v
!:2Adam/dense_115/bias/v
№2э
!__inference__wrapped_model_958013Ч
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/
conv2d_174_inputџџџџџџџџџFF
2
.__inference_sequential_59_layer_call_fn_958697
.__inference_sequential_59_layer_call_fn_958488
.__inference_sequential_59_layer_call_fn_958426
.__inference_sequential_59_layer_call_fn_958672Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_sequential_59_layer_call_and_return_conditional_losses_958647
I__inference_sequential_59_layer_call_and_return_conditional_losses_958326
I__inference_sequential_59_layer_call_and_return_conditional_losses_958599
I__inference_sequential_59_layer_call_and_return_conditional_losses_958363Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_conv2d_174_layer_call_fn_958717Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_174_layer_call_and_return_conditional_losses_958708Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
2__inference_max_pooling2d_174_layer_call_fn_958025р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Е2В
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_958019р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_dropout_134_layer_call_fn_958739
,__inference_dropout_134_layer_call_fn_958744Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_134_layer_call_and_return_conditional_losses_958734
G__inference_dropout_134_layer_call_and_return_conditional_losses_958729Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_conv2d_175_layer_call_fn_958764Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_175_layer_call_and_return_conditional_losses_958755Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
2__inference_max_pooling2d_175_layer_call_fn_958037р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Е2В
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_958031р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_dropout_135_layer_call_fn_958786
,__inference_dropout_135_layer_call_fn_958791Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_135_layer_call_and_return_conditional_losses_958781
G__inference_dropout_135_layer_call_and_return_conditional_losses_958776Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_conv2d_176_layer_call_fn_958811Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_176_layer_call_and_return_conditional_losses_958802Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
2__inference_max_pooling2d_176_layer_call_fn_958049р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Е2В
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_958043р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_dropout_136_layer_call_fn_958838
,__inference_dropout_136_layer_call_fn_958833Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_136_layer_call_and_return_conditional_losses_958828
G__inference_dropout_136_layer_call_and_return_conditional_losses_958823Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_flatten_58_layer_call_fn_958849Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_flatten_58_layer_call_and_return_conditional_losses_958844Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_114_layer_call_fn_958869Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_114_layer_call_and_return_conditional_losses_958860Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_dropout_137_layer_call_fn_958891
,__inference_dropout_137_layer_call_fn_958896Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_dropout_137_layer_call_and_return_conditional_losses_958886
G__inference_dropout_137_layer_call_and_return_conditional_losses_958881Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_115_layer_call_fn_958916Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_115_layer_call_and_return_conditional_losses_958907Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
<B:
$__inference_signature_wrapper_958523conv2d_174_inputЌ
!__inference__wrapped_model_958013
"#01BCLMAЂ>
7Ђ4
2/
conv2d_174_inputџџџџџџџџџFF
Њ "5Њ2
0
	dense_115# 
	dense_115џџџџџџџџџЖ
F__inference_conv2d_174_layer_call_and_return_conditional_losses_958708l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџFF
Њ "-Ђ*
# 
0џџџџџџџџџDD 
 
+__inference_conv2d_174_layer_call_fn_958717_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџFF
Њ " џџџџџџџџџDD Ж
F__inference_conv2d_175_layer_call_and_return_conditional_losses_958755l"#7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ"" 
Њ "-Ђ*
# 
0џџџџџџџџџ   
 
+__inference_conv2d_175_layer_call_fn_958764_"#7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ"" 
Њ " џџџџџџџџџ   Ж
F__inference_conv2d_176_layer_call_and_return_conditional_losses_958802l017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
+__inference_conv2d_176_layer_call_fn_958811_017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ@І
E__inference_dense_114_layer_call_and_return_conditional_losses_958860]BC0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ@
 ~
*__inference_dense_114_layer_call_fn_958869PBC0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџ@Ѕ
E__inference_dense_115_layer_call_and_return_conditional_losses_958907\LM/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_115_layer_call_fn_958916OLM/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЗ
G__inference_dropout_134_layer_call_and_return_conditional_losses_958729l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ"" 
p
Њ "-Ђ*
# 
0џџџџџџџџџ"" 
 З
G__inference_dropout_134_layer_call_and_return_conditional_losses_958734l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ"" 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ"" 
 
,__inference_dropout_134_layer_call_fn_958739_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ"" 
p
Њ " џџџџџџџџџ"" 
,__inference_dropout_134_layer_call_fn_958744_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ"" 
p 
Њ " џџџџџџџџџ"" З
G__inference_dropout_135_layer_call_and_return_conditional_losses_958776l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 З
G__inference_dropout_135_layer_call_and_return_conditional_losses_958781l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
,__inference_dropout_135_layer_call_fn_958786_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ 
,__inference_dropout_135_layer_call_fn_958791_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ З
G__inference_dropout_136_layer_call_and_return_conditional_losses_958823l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 З
G__inference_dropout_136_layer_call_and_return_conditional_losses_958828l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
,__inference_dropout_136_layer_call_fn_958833_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@
,__inference_dropout_136_layer_call_fn_958838_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Ї
G__inference_dropout_137_layer_call_and_return_conditional_losses_958881\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 Ї
G__inference_dropout_137_layer_call_and_return_conditional_losses_958886\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 
,__inference_dropout_137_layer_call_fn_958891O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@
,__inference_dropout_137_layer_call_fn_958896O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@Ћ
F__inference_flatten_58_layer_call_and_return_conditional_losses_958844a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџР
 
+__inference_flatten_58_layer_call_fn_958849T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџР№
M__inference_max_pooling2d_174_layer_call_and_return_conditional_losses_958019RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_174_layer_call_fn_958025RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ№
M__inference_max_pooling2d_175_layer_call_and_return_conditional_losses_958031RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_175_layer_call_fn_958037RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ№
M__inference_max_pooling2d_176_layer_call_and_return_conditional_losses_958043RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_176_layer_call_fn_958049RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
I__inference_sequential_59_layer_call_and_return_conditional_losses_958326~
"#01BCLMIЂF
?Ђ<
2/
conv2d_174_inputџџџџџџџџџFF
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ы
I__inference_sequential_59_layer_call_and_return_conditional_losses_958363~
"#01BCLMIЂF
?Ђ<
2/
conv2d_174_inputџџџџџџџџџFF
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 С
I__inference_sequential_59_layer_call_and_return_conditional_losses_958599t
"#01BCLM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџFF
p

 
Њ "%Ђ"

0џџџџџџџџџ
 С
I__inference_sequential_59_layer_call_and_return_conditional_losses_958647t
"#01BCLM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџFF
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ѓ
.__inference_sequential_59_layer_call_fn_958426q
"#01BCLMIЂF
?Ђ<
2/
conv2d_174_inputџџџџџџџџџFF
p

 
Њ "џџџџџџџџџЃ
.__inference_sequential_59_layer_call_fn_958488q
"#01BCLMIЂF
?Ђ<
2/
conv2d_174_inputџџџџџџџџџFF
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_59_layer_call_fn_958672g
"#01BCLM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџFF
p

 
Њ "џџџџџџџџџ
.__inference_sequential_59_layer_call_fn_958697g
"#01BCLM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџFF
p 

 
Њ "џџџџџџџџџУ
$__inference_signature_wrapper_958523
"#01BCLMUЂR
Ђ 
KЊH
F
conv2d_174_input2/
conv2d_174_inputџџџџџџџџџFF"5Њ2
0
	dense_115# 
	dense_115џџџџџџџџџ