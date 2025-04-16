import torch
import random
from numpy.polynomial import Polynomial
import numpy as np
import itertools
import math
import sys
from enum import Enum
from collections import defaultdict
from collections import deque
import time
from datetime import datetime
from typing import Type
    
# Default values
LogLevel=1
g_IfUnitTestAll=False
g_IfHugeInputMatrix=True
# Version Number
C_Version="Ver_20250415"
#C_LearningIterations=2048
C_LearningIterations=1024
#C_LearningIterations=256
#C_LearningIterations=64

C_MaxWeightValue=10**10
C_NumberofTrialMax=10.0**100
C_StableProbabilityThreshold=0.01
C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink=0.5
C_LimitofCreatingBranchConditions=10
C_LimitofPropagationBranches=20

g_step=10
g_SOLNetwork=None
C_RandomSeed=12
g_random=random.Random(C_RandomSeed)

##########################################################################
#
#  SOL Vector test implementation 
#       SOL: 自己組織化論理 (Self Organizing Logic)
##########################################################################

import logging
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M")


import os
# ユーザーディレクトリを取得
user_dir = os.path.expanduser("~")

# 出力ディレクトリを作成
log_dir = os.path.join(user_dir, "log_files")
os.makedirs(log_dir, exist_ok=True)

# ログファイルのパス
log_file_path = os.path.join(log_dir, 'SOLImplementation'+formatted_time+'.log')

logging.basicConfig(
    filename=log_file_path,          # 出力するファイル名
    filemode='w',                 # 'a'は追記モード、'w'は上書き
    level=logging.INFO,           # INFOレベル以上のログを記録
    format='%(message)s'
)
#    format='%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format='%(message)s')

def Log(*args, sep=' ', end='\n'):
    message = "step"+str(g_step)+" : "+sep.join(map(str, args)) + end.strip()
    logging.info(message)
    print(message)

def CalculateEntropy(P : float) -> float :
    C_MaxEntropyValue=10000000
    if P >= 1.0 or P <= 0.0:
        return 0

    LOGP=-math.log(P,2)
    LOG1_P=-math.log(1-P,2)
    if (LOGP>C_MaxEntropyValue):
        LOGP=C_MaxEntropyValue
    if (LOG1_P>C_MaxEntropyValue):
        LOG1_P=C_MaxEntropyValue
    return P*LOGP+(1-P)*LOG1_P

class diagnostics:
    def __init__(self):
        self._Nodes=0
        self._Links=0
        self._Activations=0
        self._Feedbacks=0
        self._PropagationMaskedLinks=0
        self._PropagatedLinks=0

    def StepReset(self):
        self._Activations=0
        self._Feedbacks=0
        self._PropagationMaskedLinks=0
        self._PropagatedLinks=0

    def DumpCurrentStatus(self):
        Log("  Nodes",self._Nodes,"Links",self._Links)
        Log("  Activations",self._Activations,"Feedbacks",self._Feedbacks)
        Log("  Propagated links",self._PropagatedLinks,"Masked links",self._PropagationMaskedLinks)
        
g_diagnostics=diagnostics()

class NetworkFactory:
    _instance=None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def CreateLogicLink(self,InputNode,OutputNode,P11:float,P00:float,N11:float,N00:float,OriginatedLink=None):
        NewLink=Link(InputNode,OutputNode,P11,P00,N11,N00,OriginatedLink)
        InputNode.EntryOutputLink(NewLink)
        OutputNode.EntryInputLink(NewLink)
        return NewLink

    def AddNewLink(self,AddActivation,TargetNode,LinkSign:bool) -> Type['Link']:
            # Not USING
            if AddActivation._PropagatingLink is None:
                return None
            SourceNode=AddActivation._InputNode
            if SourceNode==TargetNode:
                return None
            NewLink_N11=AddActivation.GetPropagatedN()
            NewLink_N00=AddActivation.GetPropagatingLink()._N00
            NewLink=self.CreateLogicLink(SourceNode,TargetNode,1 if LinkSign else 0,1 if LinkSign else 0,NewLink_N11,NewLink_N00)
            return NewLink

    def CreateANDNode(self,Name,NodeA,NodeB,LinkSignA:bool,LinkSignB:bool,A_N:float,B_N:float) -> Type['Node']:
            Log("Create AND node ",Name)
            NewLogic=ANDNode(Name+"AND")
            NewLink1=self.CreateLogicLink(NodeA,NewLogic,1 if LinkSignA else 0,1 if LinkSignA else 0,A_N,A_N)
            NewLink2=self.CreateLogicLink(NodeB,NewLogic,1 if LinkSignB else 0,1 if LinkSignB else 0,B_N,B_N)
            return NewLogic

    def CreateORNode(self,Name,NodeA,NodeB,LinkSignA:bool,LinkSignB:bool,A_N:float,B_N:float) -> Type['Link']:
            Log("Create OR node ",Name)
            NewLogic=ORNode(Name+"OR")
            NewLink1=self.CreateLogicLink(NodeA,NewLogic,1 if LinkSignA else 0,1 if LinkSignA else 0,A_N,A_N)
            NewLink2=self.CreateLogicLink(NodeB,NewLogic,1 if LinkSignB else 0,1 if LinkSignB else 0,B_N,B_N)
            return NewLogic

    def CreateXORNode(self,Name,NodeA,NodeB,LinkSignA:bool,LinkSignB:bool,A_N:float,B_N:float) -> Type['Link']:
            Log("Create XOR node ",Name)
            NewLogic=XORNode(Name+"XOR")
            NewLink1=self.CreateLogicLink(NodeA,NewLogic,1 if LinkSignA else 0,1 if LinkSignA else 0,A_N,A_N)
            NewLink2=self.CreateLogicLink(NodeB,NewLogic,1 if LinkSignB else 0,1 if LinkSignB else 0,B_N,B_N)
            return NewLogic
    
g_NetworkFactory=NetworkFactory()

##########################################################################
# Equations of propagation
##########################################################################
class Propagation_weight_object:

    def __init__(self,LinkWeight:float,Probability:float,IfP11:bool,ActivationObject):
        self._LinkWeight=LinkWeight
        self._Activation=ActivationObject
        self._IfP11=IfP11
        self._E=Probability
        self._EDegree=1

    # Not using?
    def PrepareForNextPropagation(self):
        pass

    def CreateClone(self) :
        NewPWO=Propagation_weight_object(self._LinkWeight,self._E,self._IfP11,self._Activation)
        return NewPWO

    def MultiplyToE(self,C:float):
        self._E*=C

    def MultiplyAndChooseToE(self,R:float,E:float,EDegree:float):
        C1=self._E*R
        C2=self._E*E
        C_MinimumProbabilityRatioToRemoveHigherDegree=100
        if abs(C1*C_MinimumProbabilityRatioToRemoveHigherDegree)<abs(C2):
            self._E=C2
            self._EDegree+=EDegree
        else:
            self._E=C1

    def FeedbackToLinkPWO(self,FeedbackValue:float,FeedbackN:float,ChangedRootActivation,PropagatedALO,IfCreateCondition:bool):
        if self._E==0:
            return
        FeedbackAddValue=FeedbackValue*self._LinkWeight
        if LogLevel>=2:
            Log("Lv2: Feedback to link. Weight ",self._LinkWeight,"AddValue",FeedbackAddValue)
        if (self._IfP11):
            self._Activation.ApplyFeedbackP11(self._E,FeedbackAddValue,FeedbackN,ChangedRootActivation,PropagatedALO,IfCreateCondition)
        else:
            self._Activation.ApplyFeedbackP00(self._E,FeedbackAddValue,FeedbackN,ChangedRootActivation,PropagatedALO,IfCreateCondition)

    def FeedbackBalancedToPWO(self,FeedbackN:float,ChangedRootActivation,PropagatedALO):
        if self._E==0:
            return
        if (self._IfP11):
            self._Activation.FeedbackBalancedP11(self._E,FeedbackN,ChangedRootActivation,PropagatedALO)
        else:
            self._Activation.FeedbackBalancedP00(self._E,FeedbackN,ChangedRootActivation,PropagatedALO)

    def ResumeFeedback(self):
        self._Activation.ResumeFeedback()

    def Dump(self):
        if (self._Activation!=None):
            PropagatingLink=self._Activation._PropagatingLink
            if (PropagatingLink!=None):
                    Log(" PWO",PropagatingLink._InputNode.getname(),"->",PropagatingLink._OutputNode.getname(),"IfP11",self._IfP11,"Weight",self._Weight)
            else:
                    Log(" PWO ","IfP11",self._IfP11,"Weight",self._Weight)

class Propagation_weight_vector:

    def __init__(self,InitialP:float):
        self._PWO_Vector=[]
        self._Equation=Polynomial([InitialP])

    def PrepareForNextPropagation(self):
        for PWO in self._PWO_Vector:
            PWO.PrepareForNextPropagation()

    def CreateClone(self) :
        NewPWV=Propagation_weight_vector(0)
        for PWO in self._PWO_Vector:
            NewPWV._PWO_Vector.append(PWO.CreateClone())
        NewPWV._Equation=self._Equation
        return NewPWV
    
    def AddRootActivation(self,PropagatedProbability,LinkWeight,IfP11,ActivationObject):
        NewObject = Propagation_weight_object(LinkWeight,PropagatedProbability,IfP11,ActivationObject)
        self._PWO_Vector.append(NewObject)
        AddP=Polynomial([0,PropagatedProbability])
        self._Equation+=AddP
   
    def _JoinPWOElements(self,AnotherPWV):
        PWO:Propagation_weight_object
        PWO2:Propagation_weight_object
        AppendingList=[]
        for PWO in AnotherPWV:
            IfFound=False
            for PWO2 in self._PWO_Vector:
                if PWO._Activation==PWO2._Activation:
                    if PWO._IfP11==PWO2._IfP11:
                        IfFound=True
                        break
            if not IfFound:
                AppendingList.append(PWO)
        self._PWO_Vector.extend(AppendingList)

    def PropagateOnActivation(self,P11R:float,P11EW:float,P00R:float,P00EW:float):
        # 1-P00+(P11+P00-1)Eq = 1-P00R-P00EWt+(P11R+P00R+P11EWt+P00EWt-1)Eq
        C1=P11R+P00R-1
        for PWO in self._PWO_Vector:
            PWO.MultiplyToE(C1)
        AddEq=Polynomial([1-P00R,-P00EW])
        MulEq=Polynomial([P11R+P00R-1,P11EW+P00EW])
        self._Equation*=MulEq
        self._Equation+=AddEq
        self._PruneEquation(self._Equation)

    def _GetPWOFromActivation(self,RefActivation):
        for P in self._PWO_Vector:
            if P._Activation==RefActivation:
                return P
        return None
        
    def _PruneEquation(self,InputEquation:Polynomial):
        # 係数の小さい次数を省略
        MaximumCoef:float = 0
        for coef in InputEquation.coef[1:]:
            if abs(coef)>abs(MaximumCoef):
                MaximumCoef=coef
        threshold = MaximumCoef*1e-4
        pruned_equation=[coef if abs(coef) > threshold else 0 for coef in InputEquation.coef]
        InputEquation.coef=pruned_equation
    
    def _GetMaxCoeffofEquation(self)->{float,int}:
        # 定数項を除いた係数から0以外のものを取得
        non_zero_terms = [(i, coef) for i, coef in enumerate(self._Equation.coef[1:], 1) if coef != 0]
        if len(non_zero_terms)==0:
            return 0,1
        Degree,Coeff = non_zero_terms[0]
        return Coeff,Degree

    def ANDOperation(self,InputPWV):
        # P'=P1*P2 = (R1+E1*t)*(R2+E2*t)
        # PWV part
        P:Propagation_weight_object
        for PWO in self._PWO_Vector:
            R1=InputPWV._Equation.coef[0]
            EW,Degree=InputPWV._GetMaxCoeffofEquation()
            PWO.MultiplyAndChooseToE(R1,EW,Degree)
        for NewPWO in InputPWV._PWO_Vector:
            IfFound=False
            for PWO in self._PWO_Vector:
                if PWO._Activation==NewPWO._Activation:
                    if PWO._IfP11==NewPWO._IfP11:
                        if NewPWO._EDegree<PWO._EDegree:
                            PWO._E=NewPWO._E
                            PWO._EDegree=NewPWO._EDegree
                        elif PWO._EDegree==NewPWO._EDegree:
                            PWO._E+=NewPWO._E
                        IfFound=True
                        break
            if not IfFound:
                PWODash=NewPWO.CreateClone()
                R1=self._Equation.coef[0]
                EW,Degree=self._GetMaxCoeffofEquation()
                PWODash.MultiplyAndChooseToE(R1,EW,Degree)
                self._PWO_Vector.append(PWODash)
        # Equation part
        #  P'=P1*P2
        P1=self._Equation
        P2=InputPWV._Equation
        self._Equation=P1*P2
        self._PruneEquation(self._Equation)
 
    def OROperation(self,InputPWV):
        # P'=P1+P2-P1*P2 = (R1+E1*t)+(R2+E2*t)-(R1+E1*t)*(R2+E2*t)
        # PWV Part
        # P'= (1-P1)*P2
        P:Propagation_weight_object
        for P in self._PWO_Vector:
            R1=1-InputPWV._Equation.coef[0]
            EW,Degree=InputPWV._GetMaxCoeffofEquation()
            EW1=-EW
            P.MultiplyAndChooseToE(R1,EW1,Degree)
        for NewPWO in InputPWV._PWO_Vector:
            IfFound=False
            for PWO in self._PWO_Vector:
                if PWO._Activation==NewPWO._Activation:
                    if PWO._IfP11==NewPWO._IfP11:
                        if NewPWO._EDegree<PWO._EDegree:
                            PWO._E=NewPWO._E
                            PWO._EDegree=NewPWO._EDegree
                        elif PWO._EDegree==NewPWO._EDegree:
                            PWO._E+=NewPWO._E

                        IfFound=True
                        break
            if not IfFound:
                PDash=NewPWO.CreateClone()
                R1=1-self._Equation.coef[0]
                EW,Degree=self._GetMaxCoeffofEquation()
                EW1=-EW
                PDash.MultiplyAndChooseToE(R1,EW1,Degree)
                self._PWO_Vector.append(PDash)
        # Equation part
        #  P'=P1+P2-P1*P2
        P1=self._Equation
        P2=InputPWV._Equation
        self._Equation=P1+P2-P1*P2
        self._PruneEquation(self._Equation)

    def XOROperation(self,InputPWV):
        # P'=P1+P2-2P1*P2 = (R1+E1*t)+(R2+E2*t)-2(R1+E1*t)*(R2+E2*t)
        # PWV Part
        # P'= (1-2*P1)*P2
        P:Propagation_weight_object
        for P in self._PWO_Vector:
            R1=1-2*(InputPWV._Equation.coef[0])
            EW,Degree=InputPWV._GetMaxCoeffofEquation()
            EW1=-2*EW
            P.MultiplyAndChooseToE(R1,EW1,Degree)
        for NewPWO in InputPWV._PWO_Vector:
            IfFound=False
            for PWO in self._PWO_Vector:
                if PWO._Activation==NewPWO._Activation:
                    if PWO._IfP11==NewPWO._IfP11:
                        if NewPWO._EDegree<PWO._EDegree:
                            PWO._E=NewPWO._E
                            PWO._EDegree=NewPWO._EDegree
                        elif PWO._EDegree==NewPWO._EDegree:
                            PWO._E+=NewPWO._E
                        IfFound=True
                        break
            if not IfFound:
                PDash=NewPWO.CreateClone()
                R1=1-2*self._Equation.coef[0]
                EW,Degree=self._GetMaxCoeffofEquation()
                EW1=-2*EW
                PDash.MultiplyAndChooseToE(R1,EW1,Degree)
                self._PWO_Vector.append(PDash)
        # Equation part
        #  P'=P1+P2-2P1*P2
        P1=self._Equation
        P2=InputPWV._Equation
        self._Equation=P1+P2-2*P1*P2
        self._PruneEquation(self._Equation)

    def _PruneEquation(self,InputEquation:Polynomial):
        # 係数の小さい次数を省略
        MaximumCoef:float = 0
        for coef in InputEquation.coef[1:]:
            if abs(coef)>abs(MaximumCoef):
                MaximumCoef=coef
        threshold = MaximumCoef*1e-4
        pruned_equation=[coef if abs(coef) > threshold else 0 for coef in InputEquation.coef]
        InputEquation.coef=pruned_equation

    def FeedbackBalanced(self,FeedbackP:float,CollidingP:float,CollidingN:float,ChangedRootActivation,PropagatedALO):
        if LogLevel>=3:
            TargetNode=PropagatedALO.GetTerminalNode()
            if TargetNode is not None:
                Log("Lv3: Feedback balanced. ",TargetNode.getname(),"Probability difference",CollidingP,"-",FeedbackP)
        for PWO in self._PWO_Vector:
            PWO.FeedbackBalancedToPWO(CollidingN,ChangedRootActivation,PropagatedALO)

    def FeedbackToCreateCondition(self,FeedbackP:float,CollidingP:float,CollidingN:float,ChangedRootActivation,PropagatedALO,IfCreateCondition:bool):
        if LogLevel>=3:
            TargetNode=PropagatedALO.GetTerminalNode()
            if TargetNode is not None:
                Log("Lv3: FeedbackStart() Lowest entropy feedback ",TargetNode.getname(),"FeedbackP" ,FeedbackP,"collidingP",CollidingP)
        # Calculate N from from each weight
        EWTotal=0
        for PWO in self._PWO_Vector:
            EWTotal+=abs(PWO._E*PWO._LinkWeight)
        
        # Low entropy link feedback
        ChosenPWO:Propagation_weight_object=None
        EW=0
        PWO:Propagation_weight_object
        for PWO in self._PWO_Vector:
            if PWO._E==0:
                continue
            C_SameProbabilityMargin=0.01
            if abs(PWO._E)+C_SameProbabilityMargin<abs(CollidingP-FeedbackP):
                continue
            FeedbackAddValue=((CollidingP-FeedbackP)/PWO._E)
            if abs(FeedbackAddValue)<0.5:
                continue
            EW=abs(PWO._E*PWO._LinkWeight)
            if EWTotal!=0:
                LocalN=(CollidingN/EWTotal)*EW
            else:
                LocalN=CollidingN
            ChosenPWO=PWO
            if (ChosenPWO._IfP11):
                if LogLevel>=3:
                    Log("Lv3: FeedbackToCreateCondition " ,ChosenPWO._Activation._InputNode.getname(),"->",ChosenPWO._Activation._OutputNode.getname())
                    Log("  CollidingP" ,CollidingP,"-P11 ",ChosenPWO._Activation._CurrentP11,"/E",ChosenPWO._E,"= AddValue",FeedbackAddValue)
                ChosenPWO._Activation.ApplyFeedbackP11(ChosenPWO._E,FeedbackAddValue,LocalN,ChangedRootActivation,PropagatedALO,IfCreateCondition)
            else:
                if LogLevel>=3:
                    Log("Lv3: FeedbackToCreateCondition " ,ChosenPWO._Activation._InputNode.getname(),"->",ChosenPWO._Activation._OutputNode.getname())
                    Log("  CollidingP" ,CollidingP,"-P00 ",ChosenPWO._Activation._CurrentP00,"/E",ChosenPWO._E,"= AddValue",FeedbackAddValue)
                ChosenPWO._Activation.ApplyFeedbackP00(ChosenPWO._E,FeedbackAddValue,LocalN,ChangedRootActivation,PropagatedALO,IfCreateCondition)

    def FeedbackToLinkPWO(self,FeedbackP:float,CollidingP:float,CalculatedT:float,CollidingN:float,ChangedRootActivation,PropagatedALO,IfCreateCondition:bool):
        # Minimum feedback
        if LogLevel>=3:
            TargetNode=PropagatedALO.GetTerminalNode()
            if TargetNode is not None:
                Log("Lv3: FeedbackStart() Random link feedback ",TargetNode.getname(),"FeedbackP" ,FeedbackP,"collidingP",CollidingP)
        PWO:Propagation_weight_object
        for PWO in self._PWO_Vector:
            PWO.FeedbackToLinkPWO(CalculatedT,CollidingN,ChangedRootActivation,PropagatedALO,IfCreateCondition)
    
    def ResumeFeedback(self):
        for WeightObject in self._PWO_Vector:
            WeightObject.ResumeFeedback()

    def Dump(self):
        Log(" PWV dump")
        for WeightObject in self._PWO_Vector:
            WeightObject.Dump()
        Log("  PWV Equation",self._Equation)

class SetCompareResult(Enum):
    NoRelation=0
    Same=1
    Contain=2
    Contained=3
    Complementary=4
    Exclusive=5
    # "Contain" means that when comparing A and B, A contains B as a set. In implementation, it means that B has more elements in AVec.

##########################################################################
# Propagation set
##########################################################################


class AVecSet:
    def __init__(self):
        self._Elements = {}  # {Node: Activation} の辞書型で管理

    def CreateClone(self):
        NewAVecSet = AVecSet()
        NewAVecSet._Elements = self._Elements.copy()  # 浅いコピー
        return NewAVecSet

    def getname(self):
        return ",".join(AE.getname() for AE in self._Elements.values())

    def IsEmpty(self):
        return not self._Elements

    def AddSetElement(self, NewActivation:"Activation"):
        OriginNode=NewActivation._OutputNode.GetOriginNode()
        self._Elements[OriginNode] = NewActivation

    def JoinElements(self, AnotherAVecSet:"AVecSet"):
        self._Elements.update(AnotherAVecSet._Elements)

    def RemoveElement(self, RemovingActivation):
        OriginNode=RemovingActivation._OutputNode.GetOriginNode()
        self._Elements.pop(OriginNode, None)

    def CompareElements(self, AnotherAVecSet:"AVecSet") -> "SetCompareResult":
        set1 = set(self._Elements.keys())
        set2 = set(AnotherAVecSet._Elements.keys())
        common_keys = set1 & set2  # 一致部分のノードを抽出
        
        CommonResult=SetCompareResult.Same
        if common_keys:
            CommonResult = self.CompareEachActivations({k: self._Elements[k] for k in common_keys},
                                                 {k: AnotherAVecSet._Elements[k] for k in common_keys})
        if set1 == set2:
            return CommonResult
        if set1 > set2:
            return self.MultiplyRelationofSets(CommonResult,SetCompareResult.Contained)
        elif set2 > set1:
            return self.MultiplyRelationofSets(CommonResult,SetCompareResult.Contain)

        return SetCompareResult.NoRelation

    def CompareEachActivations(self,elements1,elements2) -> "SetCompareResult":
        E1:Node
        E2:Node
        Result=SetCompareResult.Same
        for node in elements1:
            A1:Activation=elements1[node]
            E1=A1._OutputNode
            E1O=E1.GetOriginNode()
            E1N=E1.GetSeriesNumber()
            A2:Activation=elements2[node]
            E2=A2._OutputNode
            E2O=E2.GetOriginNode()
            E2N=E2.GetSeriesNumber()
            LocalResult=SetCompareResult.NoRelation
            A1P=A1._PropagatedProbability
            A2P=A2._PropagatedProbability
            IfComplementary=False
            if A1P>0.5 and A2P<0.5:
                IfComplementary=True
            elif A1P<0.5 and A2P>0.5:
                IfComplementary=True
            if E1==E2:
                if IfComplementary:
                    LocalResult=SetCompareResult.Complementary
                else:
                    LocalResult=SetCompareResult.Same
            elif E1O==E2O:
                if E1N==E2N:
                    if IfComplementary:
                        LocalResult=SetCompareResult.Complementary
                    else:
                        LocalResult=SetCompareResult.Same
                elif E1N>E2N:
                    # E1 SET is smaller
                    if A1P>0.5 and A2P<0.5:
                        LocalResult=SetCompareResult.Exclusive
                    elif A1P<0.5 and A2P>0.5:
                        LocalResult=SetCompareResult.NoRelation
                    else:
                        LocalResult=SetCompareResult.Contained
                else:
                    # E2 SET is smaller
                    if A1P>0.5 and A2P<0.5:
                        LocalResult=SetCompareResult.NoRelation
                    elif A1P<0.5 and A2P>0.5:
                        LocalResult=SetCompareResult.Exclusive
                    else:
                        LocalResult=SetCompareResult.Contain
            else:
                assert(False)
            Result=self.MultiplyRelationofSets(Result,LocalResult)
        return Result
    
    def MultiplyRelationofSets(self,Result1:SetCompareResult,Result2:SetCompareResult) -> "SetCompareResult":

        if Result1==SetCompareResult.Same:
            if Result2==SetCompareResult.Same:
                return SetCompareResult.Same
            elif Result2==SetCompareResult.Contain:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Contained:
                return SetCompareResult.Contained
            elif Result2==SetCompareResult.Complementary:
                return SetCompareResult.Complementary
            elif Result2==SetCompareResult.Exclusive:
                return SetCompareResult.Exclusive
        elif Result1==SetCompareResult.Contain:
            if Result2==SetCompareResult.Same:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Contain:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Contained:
                return SetCompareResult.NoRelation
            elif Result2==SetCompareResult.Complementary:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Exclusive:
                return SetCompareResult.Contain
        elif Result1==SetCompareResult.Contained:
            if Result2==SetCompareResult.Same:
                return SetCompareResult.Contained
            elif Result2==SetCompareResult.Contain:
                return SetCompareResult.NoRelation
            elif Result2==SetCompareResult.Contained:
                return SetCompareResult.Contained
            elif Result2==SetCompareResult.Complementary:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Exclusive:
                return SetCompareResult.Exclusive
        elif Result1==SetCompareResult.Complementary:
            if Result2==SetCompareResult.Same:
                return SetCompareResult.Complementary
            elif Result2==SetCompareResult.Contain:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Contained:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Complementary:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Exclusive:
                return SetCompareResult.Exclusive
        elif Result1==SetCompareResult.Exclusive:
            if Result2==SetCompareResult.Same:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Contain:
                return SetCompareResult.Contain
            elif Result2==SetCompareResult.Contained:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Complementary:
                return SetCompareResult.Exclusive
            elif Result2==SetCompareResult.Exclusive:
                return SetCompareResult.Exclusive
        return SetCompareResult.NoRelation
            
class AVec:
    
    def __init__(self):
        self._AVecSubstantialSet:AVecSet=AVecSet()
        self._AVecOuterSet:AVecSet=AVecSet()
        self._Probability=1

    def CreateClone(self):
        NewAVec=AVec()
        NewAVec._AVecSubstantialSet=self._AVecSubstantialSet.CreateClone()
        NewAVec._AVecOuterSet=self._AVecOuterSet.CreateClone()
        return NewAVec

    def getname(self):
        MatchedSetName=self._AVecSubstantialSet.getname()
        LargerSetName=self._AVecOuterSet.getname()
        return f"AVec:{MatchedSetName}({LargerSetName})"
    
    def AddSubstantialElement(self,NewActivation):
        self._AVecSubstantialSet.AddSetElement(NewActivation)

    def AddOuterElement(self,NewActivation):
        self._AVecOuterSet.AddSetElement(NewActivation)
        
    def JoinElements(self,AnotherAVec):
        AVecAnotherM = AnotherAVec._AVecSubstantialSet 
        if self._AVecSubstantialSet.IsEmpty():
            self._AVecSubstantialSet.JoinElements(AVecAnotherM)
        elif self._AVecSubstantialSet.CompareElements(AVecAnotherM)==SetCompareResult.NoRelation:
            self._AVecSubstantialSet.JoinElements(AVecAnotherM)

        AVecAnotherL = AnotherAVec._AVecOuterSet 
        if self._AVecOuterSet.IsEmpty():
            self._AVecOuterSet.JoinElements(AVecAnotherL)
        elif self._AVecOuterSet.CompareElements(AVecAnotherL) == SetCompareResult.NoRelation:
            self._AVecOuterSet.JoinElements(AVecAnotherL)

    def CompareSubstantialElements(self,AnotherAVec) -> SetCompareResult:
        # for test only now
        if self._AVecSubstantialSet is None:
            return SetCompareResult.NoRelation
        return self._AVecSubstantialSet.CompareElements(AnotherAVec._AVecSubstantialSet)

    def CompareOuterElements(self,AnotherAVec) -> SetCompareResult:
        if AnotherAVec is None:
            # This set is smaller 
            return SetCompareResult.Contained
        AnotherSet=AnotherAVec._AVecOuterSet
        if self._AVecOuterSet is None:
            if AnotherSet is None:
                return SetCompareResult.Same
            else:
                # Another set is smaller 
               return SetCompareResult.Contain
        return self._AVecOuterSet.CompareElements(AnotherSet)

    def IfOuterSetIsEmpty(self) -> bool:
        return self._AVecOuterSet.IsEmpty()

##########################################################################
# Activation on Logic Operation
##########################################################################

class ALO:

    def __init__(self,P:float,N:float,Vec:AVec,ParentActivation,ChangedRootActivation=None):
        self._JoinedProbability: float =P
        self._JoinedN : float =N
        self._JoinedInputs:int = 0
        self._AVec = Vec
        self._ChangedRootActivation = ChangedRootActivation
        self._Feedbacked=False
        self._IfUpdated=True

        if ParentActivation is not None:
            if isinstance(ParentActivation,list):
                self._ParentActivations = ParentActivation
            else:
                assert(False)
                self._ParentActivations = [ParentActivation]
        else:
            self._ParentActivations = []
        
    def GetJoinedProbability(self) -> float:
        return self._JoinedProbability
    
    def GetJoinedN(self) -> float :
        return self._JoinedN
    
    def GetMaxS(self) -> float :
        return self._MaxS
        
    def GetAVec(self):
        return self._AVec
    
    def GetParentActivations(self):
        return self._ParentActivations
    
    def GetChangedRootActivation(self):
        return self._ChangedRootActivation

    def GetTerminalActivation(self):
        if len(self._ParentActivations)==0:
            return None
        return self._ParentActivations[0]
        
    def GetTerminalLink(self):
        TerminalActivation=self.GetTerminalActivation()
        if TerminalActivation is not None:
            return TerminalActivation._PropagatingLink
        else:
            return None

    def GetCollidingNode(self):
        TerminalActivation=self.GetTerminalActivation()
        if TerminalActivation is not None:
            return TerminalActivation._OutputNode
        else:
            return None

    def SetJoinedProbability(self,P:float):
        self._JoinedProbability=P

    def SetJoinedN(self,N:float):
        self._JoinedN=N

    def SetAVec(self,NewAVec):
        self._AVec=NewAVec

    def Activate(self,NewActivation):
        self._ParentActivations.append(NewActivation)

##########################################################################
# Nodes
##########################################################################


class ConditionList:
    def __init__(self):
        self._ConditionList=[]

class NodeType(Enum):
    NotDefined=0
    Real=1
    AND=2
    OR=3
    XOR=4
    Series=5

class Node:

    def __init__(self,name):
        self._Name=name
        self._InputLinks=[]
        self._OutputLinks=[]

        self._CopySourceNode=None
        self._IfCandidateExists=False
        
        self._IfObserved=False
        self._ALO=[]
        self._PreviousALO=None
        g_diagnostics._Nodes+=1

    def Reset(self):
        self._ALO=[]
        self.PrepareForNextPropagation()

    def PrepareForNextPropagation(self):
        L:Link
        for L in self._OutputLinks:
            L.PrepareForNextPropagation()
        self._IfActivated=False
        AL:ALO
        for AL in self._ALO:
            AL._Feedbacked=False
            AL._IfUpdated=False

    def __repr__(self):
        return f"Node:(name='{self._Name}')"
    
    #
    #   Access internal data
    #

    def getname(self):
        return self._Name
    
    def EntryInputLink(self,NewLink):
        self._InputLinks.append(NewLink)

    def EntryOutputLink(self,NewLink):
        self._OutputLinks.append(NewLink)

    def RemoveInputLink(self,OldLink):
        if OldLink in self._InputLinks:
           self._InputLinks.remove(OldLink)
    
    def RemoveOutputLink(self,OldLink):
        if OldLink in self._OutputLinks:
            self._OutputLinks.remove(OldLink)

    def GetSingleOutputLink(self):
        if self._OutputLinks == []:
            return None
        return self._OutputLinks[0]
    
    def GetCurrentALO(self) -> ALO :
        if len(self._ALO)==0:
            return None
        return self._ALO[0]
    
    def GetOperationString(self) -> str:
        return ","
    
    def IfSeriesNode(self) -> bool :
        return False
    
    def IfOperationNode(self) -> bool :
        return False
    
    def IfANDNode(self) -> bool :
        return False

    def IfORNode(self) -> bool :
        return False

    def IfXORNode(self) -> bool :
        return False

    def IfRealNode(self) -> bool :
        return False

    def GetNodeType(self) -> NodeType:
        return NodeType.NotDefined
    
    def CreateCloneNode(self) :
        NewNode=Node(self.getname()+"+")
        NewNode._CopySourceNode=self
        return NewNode

    def GetCopySourceNode(self) -> "Node":
        if self._CopySourceNode is None:
            return self
        return self._CopySourceNode.GetCopySourceNode()
    
    def GetOriginNode(self) -> "Node":
        return self

    def GetSeriesNumber(self) -> int:
        return 0
    
    def SetIfObserved(self,Flag:bool):
        self._IfObserved=Flag

    #
    #   Value propagating section
    #
    def SubstituteValue(self,Probability:float,N:float):
        if self._ALO==[]:
            NewALO=ALO(Probability,N,AVec(),[])
            NewALO._JoinedInputs=1
            self._ALO.append(NewALO)
        else:
            CurrentALO=self._ALO[0]
            CurrentN=CurrentALO._JoinedN
            # Modify ALO if N is larger
            CurrentS=CalculateEntropy(CurrentALO._JoinedProbability)
            NewS=CalculateEntropy(Probability)
            IfReplace=False
            if (CurrentS>NewS):
                # MinimumEntropy
                IfReplace=True
            elif NewS==CurrentS:
                if (CurrentN<N):
                    # Maximum N
                    IfReplace=True
            if IfReplace:
                CurrentALO._JoinedProbability=Probability
                CurrentALO._JoinedN=N
                CurrentALO._JoinedInputs=1

    def GetJoinedProbability(self):
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None:
            return 0.5
        return CurrentALO.GetJoinedProbability()

    def GetJoinedN(self):
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None:
            return 0
        return CurrentALO.GetJoinedN()
    
    def GetValue(self):
        return self.GetJoinedProbability()
        
    def PropagateValue(self):
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None or CurrentALO._JoinedInputs==0:
            return
        JoinedProbability=CurrentALO.GetJoinedProbability()
        JoinedN=CurrentALO.GetJoinedN()
        L:Link
        for L in self._OutputLinks:
            if L._IfCandidate:
                L.PropagateValue(JoinedProbability,JoinedN)

    #
    #   Activating section
    #
    def ActivateRoot(self,GlobalRootNode,RootProbability:float,RootN:float) :
        NewAVec=AVec()
        RootActivation=Activation(RootProbability,RootProbability,RootN,None,GlobalRootNode,self,[],NewAVec)
        NewAVec.AddSubstantialElement(RootActivation)
        NewALO=ALO(RootProbability,C_NumberofTrialMax,NewAVec,[RootActivation])
        NewALO._JoinedInputs+=1
        self._ALO=[NewALO]
        return RootActivation

    def ActivateObservation(self,PropagatedProbability:float,ParentActivation,IfPropagateOnlyChanged=False,ComplementaryRootActivation=None):
        # Check fluctuation
        IfFluctuated=False
        PreviousObservation=None
        if self._PreviousALO is not None:
            ParentActivations=self._PreviousALO._ParentActivations
            if len(ParentActivations)==1:
                ExistingActivation=ParentActivations[0]
                if ExistingActivation._PropagatingLink is None:
                    PreviousObservation=ExistingActivation

        if IfPropagateOnlyChanged and PreviousObservation is not None:
            # Check value fluctuation
            PreviousPropagatedProbability=PreviousObservation.GetPropagatedProbability()
            C_Difference=0.01
            if abs(PropagatedProbability-PreviousPropagatedProbability)>C_Difference:
                IfFluctuated=True
            if not IfFluctuated:
                return
        
        self.RemovePreviousObservations()
        RootN=ParentActivation.GetPropagatedN()
        RootProbability=ParentActivation.GetPropagatedProbability()
        RootAVec:AVec=ParentActivation.GetAVec().CreateClone()
        RootNode=ParentActivation._OutputNode
        # Probability equation starts from observaiton target node
        ActivationObservation=Activation(RootProbability,PropagatedProbability,RootN,None,RootNode,self,[],RootAVec,PreviousObservation,ComplementaryRootActivation)

        # Activation
        RootAVec.AddOuterElement(ActivationObservation)
        NewN=RootN
        NewALO=ALO(PropagatedProbability,NewN,RootAVec,[ActivationObservation],ComplementaryRootActivation)
        NewALO._JoinedInputs+=1
        self._ALO.append(NewALO)

        #  Candidate for condition
        if IfFluctuated:
            g_SOLNetwork._AssociationFactory.EntryDifferencialObservationALO(NewALO)

        self._PreviousALO=NewALO
        return ActivationObservation

    def RemovePreviousObservations(self):
        for AL in self._ALO:
            ParentActivations=AL.GetParentActivations()
            if len(ParentActivations)==1:
                ExistingActivation=ParentActivations[0]
                if ExistingActivation._PropagatingLink is None:
                    self._ALO.remove(AL)


    def AddActivation(self,CurrentALO:ALO,InputActivation):
        # Virtual method. Defined in derived class
        pass

    def ActivateNode(self,InputActivation):
        if self.IfOperationNode():
            # Evaluate later in Node.JoinActivations()
            pass
        else:
            Probability=InputActivation.GetPropagatedProbability()
            N=InputActivation.GetPropagatedN()
            OldAVec=InputActivation.GetAVec()
            NewAVec=OldAVec.CreateClone()
            ChangedRootActivation=InputActivation.GetChangedRootActivation()

            InputNode:Node=InputActivation._InputNode
            CurrentALO:ALO=None
            CA:ALO
            for CA in self._ALO:
                ParentActivations=CA.GetParentActivations()
                if len(ParentActivations)==1:
                    AA:Activation=ParentActivations[0]
                    if AA._InputNode==InputNode:
                        CurrentALO=CA

            if CurrentALO is not None:
                CurrentALO._JoinedProbability=Probability
                CurrentALO._JoinedN=N
                CurrentALO._AVec=NewAVec
                CurrentALO._ParentActivations=[InputActivation]
                CurrentALO._ChangedRootActivation=ChangedRootActivation
                CurrentALO._IfUpdated=True
            else:
                NewALO=ALO(Probability,N,NewAVec,[InputActivation],ChangedRootActivation)
                NewALO._IfUpdated=True
                NewALO._JoinedInputs=1
                self._ALO.append(NewALO)
                CurrentALO=NewALO

    def JoinActivations(self,CurrentALO:ALO):
        assert(self.IfOperationNode())
        # Initialize ALO
        IL:Link 
        for IL in self._InputLinks:
            if not IL._IfPropagate:
                continue
            CA=IL._CurrentActivation
            if CA is not None:
                self.AddActivation(CurrentALO,CA)

        self.ApplyMissingInputs(CurrentALO)

        self._PreviousALO=CurrentALO

    def CheckAVecRelation(self,CurrentAVec:AVec,A2) -> bool:
        # Compare outer sets of each AVec
        if CurrentAVec.IfOuterSetIsEmpty():
            # AVec is empty. Always true.
            return True
        # Check AVec
        CompareAVec=A2.GetAVec()
        # Coompare AVec
        if CompareAVec is None:
            return True
        if CompareAVec.IfOuterSetIsEmpty():
            # AVec is empty. Always true.
            return True
        CompareResult=CurrentAVec.CompareOuterElements(CompareAVec)
        if not CompareResult==SetCompareResult.NoRelation:
            # Condition already included
            return False
        return True

    def CheckPropagationReady(self,CurrentALO:ALO) -> bool:
        return True

    def ApplyMissingInputs(self,CurrentALO:ALO):
        pass

    def PropagateActivation(self) :
        # propagate to output links
        # Called from SOLNetwrk learning loop
        if len(self._ALO)==0:
            return
        ChosenALO=None
        if self.IfOperationNode():
            ### Choose 1 ALO from ALOVector
            ChosenALO=self._ALO[0]
            self.JoinActivations(ChosenALO)
            if len(ChosenALO._ParentActivations)==0:
                # Not activated
                return 
        else:
            ChosenALO=self.CheckNodeCollision2()

        if ChosenALO is None:
            return
        if not self.CheckPropagationReady(ChosenALO):
            #  Inputs are not enough
            return

        if LogLevel>=3:
            Log("Lv3: Propagating Activation",self.getname(),"Probability",ChosenALO.GetJoinedProbability(),"N",ChosenALO.GetJoinedN())

        L:Link
        # Choose links to propagate
        C_MaxNumberofLinkPropagations=10000
        NumberOfMaxPropagations=0
        RandomLinks=0
        for L in self._OutputLinks:
            if L._IfRandom:
                RandomLinks+=1
            elif NumberOfMaxPropagations<C_MaxNumberofLinkPropagations:
                if L._IfPropagate:
                    L.ActivateLink(ChosenALO)
                    NumberOfMaxPropagations+=1
                    g_diagnostics._PropagatedLinks+=1
                else:
                    g_diagnostics._PropagationMaskedLinks+=1

        if RandomLinks>10:
            # Clean up
            TemporaryList=[]
            C_NThresholdofRemovingLink=5
            for L in self._OutputLinks:
                if not L._IfRandom or L.GetN11()<C_NThresholdofRemovingLink or L.GetN00()<C_NThresholdofRemovingLink :
                    TemporaryList.append(L)
            self._OutputLinks=TemporaryList

    def JoinPWV(self,OldPWV:Propagation_weight_vector,NewPWV:Propagation_weight_vector) -> Propagation_weight_vector:
        return NewPWV.CreateClone()


    #
    #   Feedback section
    #
    def CheckNodeCollision2(self) -> ALO :
        # @Return true -> feedback applied

        ChosenALO:ALO=None
        CA:ALO=None    
        # Choose reference ALO
        for CA in self._ALO :
            if CA._Feedbacked:
                continue
            ParentActivations=CA._ParentActivations
            if len(ParentActivations)==1:
                A:Activation=ParentActivations[0]
                InputLink=A._PropagatingLink
                if InputLink is not None and not InputLink._IfPropagate:
                    continue
            CP=CA.GetJoinedProbability()
            if CalculateEntropy(CP)>CalculateEntropy(C_StableProbabilityThreshold):
                continue
            CN=CA._JoinedN
            if ChosenALO is None:
                ChosenALO=CA
            elif CN>ChosenALO._JoinedN:
                ChosenALO=CA

        if ChosenALO is None:
            return None
        
        ReturnALO=ChosenALO
        KeepingALOs=[]
        for CAL in self._ALO :
            CPA=CAL.GetParentActivations()
            if len(CPA)==1:
                CA:Activation
                CA=CPA[0]
                PropagatingLink=CA._PropagatingLink
                if PropagatingLink is not None:
                    if PropagatingLink._IfRandom or not PropagatingLink._IfPropagate:
                        continue
            KeepingALOs.append(CAL)
            if CAL==ChosenALO:
                continue                
            if not CAL._IfUpdated and not ChosenALO._IfUpdated:
                continue
            if not CAL._Feedbacked:
                CN=CAL._JoinedN
                if CN<ChosenALO._JoinedN:
                    if not self.CheckNodeCollision(CAL,ChosenALO):
                        if ReturnALO is None:
                            ReturnALO=CAL
                        elif ReturnALO._JoinedN<CN:
                            ReturnALO=CAL
            CAL._IfUpdated=False
        
        if len(KeepingALOs)<len(self._ALO):
            self._ALO=KeepingALOs

        ChosenALO._IfUpdated=False

        return ReturnALO

    def CheckNodeCollision(self,CurrentALO:ALO,CollidingALO:ALO) -> bool :
        # @Return true -> feedback applied
        if not self._ALO:
            return False
        if CollidingALO is None:
            return
        FeedbackP:float=CurrentALO._JoinedProbability
        FeedbackAVec:AVec=CurrentALO.GetAVec()
        FeedbackFrontLink:Link=CurrentALO.GetTerminalLink()
        IfStopPropagation=False

        CA:ALO=CollidingALO
        CollidingP=CA._JoinedProbability
        CAVEC=CA._AVec
        C_CRA=CA._ChangedRootActivation
        CollidingN=1 # learning
        IfApplyFeedback=True
        AVecCompareResult=FeedbackAVec.CompareSubstantialElements(CAVEC)
        if AVecCompareResult==SetCompareResult.Exclusive or AVecCompareResult==SetCompareResult.Complementary:
            IfApplyFeedback=False
        if IfApplyFeedback:
            TerminalActivation=CurrentALO.GetTerminalActivation()
            FeedbackPWV=TerminalActivation.PropagatePWVRecursively()
            if FeedbackFrontLink is not None:
                IfCreateCondition=FeedbackFrontLink._IfCreateCondition
            # Start feedback and creating condition nodes
            if (abs(CollidingP-FeedbackP)<0.001):
                FeedbackPWV.FeedbackBalanced(FeedbackP,CollidingP,CollidingN,C_CRA,CurrentALO)
                # No feedback
            else:
                # Feedback start
                IfSignReversed=False
                FeedbackPDash=FeedbackP
                if FeedbackPDash>=0.5 and CollidingP<0.5:
                    IfSignReversed=True
                elif FeedbackPDash<=0.5 and CollidingP>0.5:
                    IfSignReversed=True

                if IfSignReversed:
                    # Low entropy link feedback
                    # Reverse probability of lower weight link
                    FeedbackPWV.FeedbackToCreateCondition(FeedbackPDash,CollidingP,CollidingN,C_CRA,CurrentALO,IfCreateCondition)
                else:
                    # Higher entropy feedback .. Solve polynomial equaitone
                    FeedbackEquation=FeedbackPWV._Equation
                    CalculatedT=self.CalculateFeedback(FeedbackEquation,CollidingP)
                    if CalculatedT is None:
                        assert(False)
                    elif CalculatedT!=0:
                        FeedbackPWV.FeedbackToLinkPWO(FeedbackP,CollidingP,CalculatedT,CollidingN,C_CRA,CurrentALO,IfCreateCondition)

                if FeedbackAVec is not None:
                    # Stop propagating if colliding AVEC set is larger
                    CompareResult=FeedbackAVec.CompareOuterElements(CAVEC)
                    if CompareResult==SetCompareResult.Contained:
                        IfStopPropagation=True

            CurrentALO._Feedbacked=True
            self.ApplyFeedbackResulttoNetwork(CurrentALO,CollidingP,CollidingN)

        # true -> Collision detected and feedbacked. Stop propagation.
        return IfStopPropagation

    def ApplyFeedbackResulttoNetwork(self,CALO,CollidingP:float,CollidingN:float):
        # Check feedback result and choose lowest entropy path
        A:Activation
        CurrentALO:ALO=CALO
        for A in CurrentALO._ParentActivations:
            PropagatingLink:Link
            PropagatingLink=A.GetPropagatingLink()
            if PropagatingLink is not None:
                PropagatedP=CurrentALO._JoinedProbability
                N1Add=PropagatedP*CollidingN
                P1Dash=abs(PropagatedP-CollidingP)*N1Add
                PropagatingLink._PathFeedbackP1+=P1Dash
                PropagatingLink._PathFeedbackN1+=N1Add
                N0Add=(1-PropagatedP)*CollidingN
                P0Dash=abs(PropagatedP-CollidingP)*N0Add
                PropagatingLink._PathFeedbackP0+=P0Dash
                PropagatingLink._PathFeedbackN0+=N0Add

                if PropagatingLink._IfRandom:
                    Log("IfRandomLink",A._InputNode.getname(),"->",self.getname())

    def PrepareNextPropagation(self):
        LI:Link
        IfCandidateExists=False
        FeedbackMinP1=C_NumberofTrialMax
        FeedbackMinP0=C_NumberofTrialMax
        NumberofValidLinks=0
        C_ZeroP=0.000001
        for LI in self._InputLinks:
            if FeedbackMinP1>LI._PathFeedbackP1:
                FeedbackMinP1=LI._PathFeedbackP1
            if FeedbackMinP0>LI._PathFeedbackP0:
                FeedbackMinP0=LI._PathFeedbackP0
            if  LI._PathFeedbackN1>0 and  LI._PathFeedbackN0>0 and LI._PathFeedbackP1<C_ZeroP and LI._PathFeedbackP0<C_ZeroP:
                IfCandidateExists=True
                CandidateLink=LI
            if not LI._IfRandom:
                NumberofValidLinks+=1

        for LI in self._InputLinks:
            IfCandidate=False
            IfPropagate=False
            IfCreateCondition=False
            IfRandomLink=False

            OriginatedLink=LI._OriginatedLink
            if OriginatedLink is None:
                if LI._PathFeedbackN1==0 or LI._PathFeedbackN0==0:
                    # Not tested
                    IfPropagate=True
                if LI._PathFeedbackP1<C_ZeroP and LI._PathFeedbackP0<C_ZeroP:
                    # Candidate
                    IfCandidate=True
                    IfPropagate=True
                    IfCreateCondition=True
                elif LI._PathFeedbackN1>0 and LI._PathFeedbackP1<=FeedbackMinP1:
                    # Minimum feedbacked P11
                    IfCandidate=True
                    IfPropagate=True
                    IfCreateCondition=True
                elif LI._PathFeedbackN0>0 and LI._PathFeedbackP0<=FeedbackMinP0:
                    # Minimum feedbacked P11
                    IfCandidate=True
                    IfPropagate=True
                    IfCreateCondition=True
                else:
                    if not LI._IfRandom:
                        if g_random.randint(0,NumberofValidLinks) <= C_LimitofPropagationBranches:  # M/N の確率でリストを処理
                            IfPropagate=True
                        if not IfCandidateExists:
                            # Candidate for creating condition
                            if g_random.randint(0,NumberofValidLinks) <= C_LimitofCreatingBranchConditions:  # M/N の確率でリストを処理
                                IfPropagate=True
                                IfCreateCondition=True
            else:
                # Choose certain link 
                IfCheckCondition=False
                if LI._PathFeedbackP1<C_ZeroP and LI._PathFeedbackP0<C_ZeroP:
                    IfCheckCondition=True
                    IfCandidate=True
                elif IfCandidateExists:
                    pass
                else:
                    if LI._PathFeedbackN1==0 or LI._PathFeedbackN0==0:
                        # Not tested
                        IfPropagate=True
                    elif LI._PathFeedbackN1>0 and LI._PathFeedbackP1<=FeedbackMinP1:
                        # Minimum feedbacked P11
                        IfCheckCondition=True
                    elif LI._PathFeedbackN0>0 and LI._PathFeedbackP0<=FeedbackMinP0:
                        # Minimum feedbacked P00
                        IfCheckCondition=True
                    else:
                        if not LI._IfRandom:
                            if g_random.randint(0,NumberofValidLinks) <= C_LimitofPropagationBranches:  # M/N の確率でリストを処理
                                IfPropagate=True
                            if not IfCandidateExists:
                                # Candidate for creating condition
                                if g_random.randint(0,NumberofValidLinks) <= C_LimitofCreatingBranchConditions:  # M/N の確率でリストを処理
                                    IfPropagate=True
                                    IfCreateCondition=True
                        
                if IfCheckCondition:
                    # Derived condition link
                    C_NThresholdForAddingCondition=1
                    if LI._PathFeedbackN1>=C_NThresholdForAddingCondition and LI._PathFeedbackN0>=C_NThresholdForAddingCondition:
                        FeedbackTotalN=LI._PathFeedbackP1+LI._PathFeedbackP0
                        FeedbackOriginalTotalN=(OriginatedLink._PathFeedbackP1-LI._PathFeedbackP1Original)+(OriginatedLink._PathFeedbackP0-LI._PathFeedbackP0Original)
                        C_FeedbackNMargin=1
                        if FeedbackTotalN>FeedbackOriginalTotalN+C_FeedbackNMargin:
                            # This link is more random than the original link. Not valid.
                            IfRandomLink=True
                            IfCandidate=False
                            IfPropagate=False
                        elif FeedbackTotalN<FeedbackOriginalTotalN:
                            # Feedback of this link decreased from the original link. Valid link for more condition.
                            IfCreateCondition=True
                            IfPropagate=True
                        else:
                            #IfCreateCondition=True
                            IfPropagate=True
                    else:
                        IfPropagate=True
                if IfPropagate:
                    if not OriginatedLink._IfPropagate:
                        OriginatedLink._IfPropagate=True
                        OriginatedLink._IfPropagationUpdated=True

            if LI._IfCandidate is not IfCandidate:
                LI._IfCandidate=IfCandidate
                LI._IfPropagationUpdated=True
            if LI._IfPropagate is not IfPropagate:
                LI._IfPropagate=IfPropagate
                LI._IfPropagationUpdated=True
            if LI._IfCreateCondition is not IfCreateCondition:
                LI._IfCreateCondition=IfCreateCondition
                LI._IfPropagationUpdated=True
            if LI._IfRandom is not IfRandomLink:
                LI._IfRandom=IfRandomLink
                LI._IfPropagationUpdated=True
            
        self._IfCandidateExists=IfCandidateExists
        for LI in self._InputLinks:
            if LI._IfPropagationUpdated:
                LI._InputNode.PrepareNextPropagationRecursively(IfCandidate,IfPropagate,IfCreateCondition,IfRandomLink)
            LI._IfPropagationUpdated=False

    def PrepareNextPropagationRecursively(self,IfCandidate:bool,IfPropagate:bool,IfCreateCondition:bool,IfRandom:bool):
        for LI in self._OutputLinks:
            if LI._IfCandidate:
                IfCandidate=True
            if LI._IfPropagate:
                IfPropagate=True
            if LI._IfCreateCondition:
                IfCreateCondition=True
            if not LI._IfRandom:
                IfRandom=False
        LI:Link
        for LI in self._InputLinks:
            Update=False
            if LI._IfCandidate is not IfCandidate:
                Update=True
                LI._IfCandidate=IfCandidate
            if LI._IfPropagate is not IfPropagate:
                Update=True
                LI._IfPropagate=IfPropagate
            if LI._IfCreateCondition is not IfCreateCondition:
                Update=True
                LI._IfCreateCondition=IfCreateCondition
            if LI._IfRandom is not IfRandom:
                Update=True
                LI._IfRandom=IfRandom
            if Update:
                LI._InputNode.PrepareNextPropagationRecursively(IfCandidate,IfPropagate,IfCreateCondition,IfRandom)

    def CalculateFeedback(self,Equation:Polynomial,CollidingP:float):
        WeightPolynomialEquation=Equation
        PropagatedP=WeightPolynomialEquation.coef[0]
        WeightPolynomialEquation.coef[0]= WeightPolynomialEquation.coef[0]-CollidingP

        if len(WeightPolynomialEquation.coef)<2:
            return 0
        if (WeightPolynomialEquation.coef[1]!=0):
            SingleDegreeResult=-(WeightPolynomialEquation.coef[0]/WeightPolynomialEquation.coef[1])
        else:
            if len(WeightPolynomialEquation.coef)==2:
                # Impossible to solve
                return 0
            SingleDegreeResult=1
        try:
            roots=WeightPolynomialEquation.roots()
        except np.linalg.LinAlgError as e:
            #Log(f"Error occurred: {e}")
            #Log("Equation",WeightPolynomialEquation.coef)
            return 0

        if len(roots)<=0:
            #Log("Invalid result ",ResultList1)
            return 0

        # 最小の実数根を取得
        CalculatedT=sys.maxsize
        for A in roots:
            AReal=A.real
            if SingleDegreeResult>0 and AReal>0:
                if abs(CalculatedT)>abs(A.imag):
                    CalculatedT=AReal
            if SingleDegreeResult<0 and AReal<0:
                if abs(CalculatedT)>abs(A.imag):
                    CalculatedT=AReal
        
        if CalculatedT==0:
            if LogLevel>=1:
                Log("Lv1: Skip feedback due to calcuration error. P Difference" ,self._Equation.coef[0]-CollidingP ,"WeightPolynomialEquation",WeightPolynomialEquation," Final delta T==0")
                Log("Lv1: Final result list ",roots)
                Log("Lv1: CalculatedT " ,CalculatedT)
                Log("Lv1: Propagating PWV is below")
                self.Dump()
            return 0

        if LogLevel>=2:
            Log("Lv2: Apply feedback P to feedback " ,PropagatedP,"observedP",CollidingP ,"WeightPolynomialEquation",WeightPolynomialEquation," Final delta T",CalculatedT)

        return CalculatedT

#
#  Others
#
    def DumpEquationTree(self,Depth=0,IfAll=False,IfRoot=False) -> tuple[str,float,float,float] :
        mes=""
        EntropyMin=0.0
        EntropyMax=0.0
        ReturnN=0
        #
        IfTerminal=(not IfRoot) and self._IfObserved
        if len(self._InputLinks)==0 or IfTerminal:
            mes=self.getname()
            EntropyMin=CalculateEntropy(1.0)
            EntropyMax=CalculateEntropy(1.0)
            return mes,EntropyMin,EntropyMax,C_NumberofTrialMax
        elif self.IfOperationNode():
            EntropyMin=CalculateEntropy(0.5)
            EntropyMax=CalculateEntropy(1.0)
            mes="("
            IfFirst=True
            ReturnN=C_NumberofTrialMax
            NumofEq=0
            L:Link
            for L in self._InputLinks:
                LinkEq,EMin,EMax,ParentNMin=L.DumpEquationTree(Depth,IfAll,False)
                if LinkEq =="":
                    continue
                if (EMin<=EntropyMin):
                    EntropyMin=EMin
                if (EMax>EntropyMax):
                    EntropyMax=EMax
                if ParentNMin<ReturnN:
                    ReturnN=ParentNMin
                if IfFirst:
                    IfFirst=False
                else:
                    mes+=self.GetOperationString()
                mes+=LinkEq
                NumofEq+=1
            mes+=")"
            if NumofEq==0:
                return "",EntropyMin,EntropyMax,ReturnN
            else:
                return mes,EntropyMin,EntropyMax,ReturnN
        else: #Joint node ... Choose least entropy
            IfFirst=True
            NumofEq=0
            C_EntropyMinimum=CalculateEntropy(0.9999999)
            ReturnN=0
            CEMin=CEMax=MinimumEntropy=CalculateEntropy(0.5)
            mes2=""
            L:Link
            for L in self._InputLinks:
                LinkEq,EMin,EMax,CurrentN=L.DumpEquationTree(Depth,IfAll,False)
                if LinkEq=="":
                    continue
                #IfReady=L.IfReadyForDumpEquation()
                IfReady=L._IfCandidate
                if (IfReady or IfAll) and CurrentN>0:
                        if (IfFirst):
                            IfFirst=False
                        else:
                            mes+="||"
                        mes+=LinkEq
                        mes+="{N="+str(CurrentN)+"}"
                        NumofEq+=1
                        if (ReturnN<CurrentN):
                            ReturnN=CurrentN

            if NumofEq==0:
                if mes2!="":
                    return mes2,CEMin,CEMax,ReturnN
                else:
                    # return Nothing
                    return "",EntropyMin,EntropyMax,ReturnN
            elif NumofEq==1:
                return mes,EntropyMin,EntropyMax,ReturnN
            else:
                return "["+mes+"]",EntropyMin,EntropyMax,ReturnN

    def PrepareForTestPropagation(self):
        for L in self._OutputLinks:
            L._IfCandidate=True

class RealNode(Node):

    def __init__(self,name):
        super().__init__(name)
        self.PrepareForNextPropagation()

    def __repr__(self):
        return f"RealNode:(name='{self._Name}')"

    def PrepareForNextPropagation(self):
        super().PrepareForNextPropagation()

    def IfRealNode(self) -> bool :
        return True

    def GetNodeType(self) -> NodeType:
        return NodeType.Real

    def CreateCloneNode(self) :
        NewNode=RealNode(self.getname()+"+")
        NewNode._CopySourceNode=self.GetCopySourceNode()
        return NewNode

class SeriesNode(Node):
    def __init__(self,OriginNode:Node,Number:int):

        NameSeries=OriginNode._Name+"."+str(Number)
        super().__init__(NameSeries)
        self._OriginNode=OriginNode
        self._SeriesNumber=Number
        self.PrepareForNextPropagation()

    def __repr__(self):
        return f"SeriesNode:(name='{self._Name}')"

    def CreateCloneNode(self) -> "Node":
        NewNode=super().CreateCloneNode()
        NewNode._OriginNode=self._OriginNode
        NewNode._SeriesNumber=self._SeriesNumber
        return NewNode
    
    def PrepareForNextPropagation(self):
        super().PrepareForNextPropagation()

    def IfSeriesNode(self) -> bool :
        return True

    def GetNodeType(self) -> NodeType:
        return NodeType.Series

    def GetOriginNode(self) -> "Node":
        return self._OriginNode
    
    def GetSeriesNumber(self) -> int:
        return self._SeriesNumber

    def CreateCloneNode(self) :
        NewNode=SeriesNode(self.getname()+"+")
        NewNode._CopySourceNode=self.GetCopySourceNode()
        return NewNode
    
class ANDNode(Node):

    def __init__(self,name):
        super().__init__(name)
        self.PrepareForNextPropagation()

    def __repr__(self):
        return f"ANDNode:(name='{self._Name}')"

    def PrepareForNextPropagation(self):
        super().PrepareForNextPropagation()
        NewALO=ALO(1,C_NumberofTrialMax,AVec(),[])
        self._ALO=[NewALO]

    #
    #   Access internal data
    #

    def GetOperationString(self) -> str:
        return " AND "

    def IfOperationNode(self) -> bool :
        return True

    def IfANDNode(self) -> bool :
        return True

    def CreateCloneNode(self) :
        NewNode=ANDNode(self.getname()+"+")
        NewNode._CopySourceNode=self.GetCopySourceNode()
        return NewNode
    
    def GetNodeType(self) -> NodeType:
        return NodeType.AND

    def SubstituteValue(self,P:float,N:float):
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None:
            return
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=OldProbability*P
        CurrentALO.SetJoinedProbability(NewProbability)
        if (CurrentALO.GetJoinedN()>N):
            CurrentALO.SetJoinedN(N)
        CurrentALO._JoinedInputs+=1

    def AddActivation(self,CurrentALO:ALO,NewInputActivation):
        # Substitute to node inputs
        # P'=P*NewP
        if (NewInputActivation==None):
            return
        if not self.CheckAVecRelation(CurrentALO.GetAVec(),NewInputActivation):
            return

        PropagatedProbability=NewInputActivation.GetPropagatedProbability()
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=OldProbability*PropagatedProbability
        CurrentALO.SetJoinedProbability(NewProbability)
 

        NewN=NewInputActivation.GetPropagatedN()
        if (CurrentALO.GetJoinedN()>NewN):
            CurrentALO.SetJoinedN(NewN)

        CurrentALO._ParentActivations.append(NewInputActivation)
        CurrentALO._AVec.JoinElements(NewInputActivation.GetAVec())
        ChangedRootActivation=NewInputActivation.GetChangedRootActivation()
        if ChangedRootActivation is not None and CurrentALO._ChangedRootActivation is None:
            CurrentALO._ChangedRootActivation=ChangedRootActivation
        CurrentALO._JoinedInputs+=1

        if LogLevel>=3:
            Log("Lv3: AND operation : Joined ALO for Activation",self.getname())
            Log("Lv3:   OldProbability",OldProbability,"ApplyingProbability",PropagatedProbability,"JoinedProbability",NewProbability)

    def JoinPWV(self,OldPWV:Propagation_weight_vector,NewPWV:Propagation_weight_vector) -> Propagation_weight_vector:
        if (OldPWV==None):
            OldPWV=NewPWV.CreateClone()
        else:
            OldPWV.ANDOperation(NewPWV)
        return OldPWV

    def CheckPropagationReady(self,CurrentALO:ALO) -> bool:
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return True
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=OldProbability*0.5
        C_SameEntropyMargin=0.001
        if CalculateEntropy(NewProbability)<=CalculateEntropy(OldProbability)+C_SameEntropyMargin:
            return True
        return False

    def ApplyMissingInputs(self,CurrentALO:ALO):
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=OldProbability*0.5
        CurrentALO.SetJoinedProbability(NewProbability)


class ORNode(Node):

    def __init__(self,name):
        super().__init__(name)
        self.PrepareForNextPropagation()

    def __repr__(self):
        return f"ORNode:(name='{self._Name}')"

    def PrepareForNextPropagation(self):
        super().PrepareForNextPropagation()
        NewALO=ALO(0,C_NumberofTrialMax,AVec(),[])
        self._ALO=[NewALO]

    #
    #   Access internal data
    #

    def GetOperationString(self) -> str:
        return " OR "

    def IfOperationNode(self) -> bool :
        return True

    def IfORNode(self) -> bool :
        return True

    def CreateCloneNode(self) :
        NewNode=ORNode(self.getname()+"+")
        NewNode._CopySourceNode=self.GetCopySourceNode()
        return NewNode
        
    def GetNodeType(self) -> NodeType:
        return NodeType.OR

    def SubstituteValue(self,P:float,N:float):
        # P' = 1-(1-P)(1-NewP)= P+NewP-PNewP
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None:
            return
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=P*(1-OldProbability)
        NewProbability+=OldProbability
        CurrentALO.SetJoinedProbability(NewProbability)
        if (CurrentALO.GetJoinedN()>N):
            CurrentALO.SetJoinedN(N)
        CurrentALO._JoinedInputs+=1

    def AddActivation(self,CurrentALO:ALO,NewInputActivation):
        # Substitute to node inputs
        # P' = 1-(1-P)(1-NewP)= P+NewP-PNewP
        if (NewInputActivation==None):
            return
        if not self.CheckAVecRelation(CurrentALO.GetAVec(),NewInputActivation):
            return

        PropagatedProbability=NewInputActivation.GetPropagatedProbability()
        OldProbability=CurrentALO.GetJoinedProbability()

        P1=1-PropagatedProbability
        P1*=(1-OldProbability)
        NewProbability=1-P1
        CurrentALO.SetJoinedProbability(NewProbability)

        NewN=NewInputActivation.GetPropagatedN()
        if (CurrentALO.GetJoinedN()>NewN):
            CurrentALO.SetJoinedN(NewN)

        CurrentALO._ParentActivations.append(NewInputActivation) 
        CurrentALO._AVec.JoinElements(NewInputActivation.GetAVec())
        ChangedRootActivation=NewInputActivation.GetChangedRootActivation()
        if ChangedRootActivation is not None and CurrentALO._ChangedRootActivation is None:
            CurrentALO._ChangedRootActivation=ChangedRootActivation
        CurrentALO._JoinedInputs+=1
 
        if LogLevel>=3:
            Log("Lv3: OR operation : Joined ALO for Activation",self.getname())
            Log("Lv3:   OldProbability",OldProbability,"ApplyingProbability",PropagatedProbability,"JoinedProbability",NewProbability)

    def JoinPWV(self,OldPWV:Propagation_weight_vector,NewPWV:Propagation_weight_vector) -> Propagation_weight_vector:
        if (OldPWV==None):
            OldPWV=NewPWV.CreateClone()
        else:
            OldPWV.OROperation(NewPWV)
        return OldPWV
    
    def CheckPropagationReady(self,CurrentALO:ALO) -> bool:
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return True
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=0.5*(1-OldProbability)
        NewProbability+=OldProbability
        C_SameEntropyMargin=0.001
        if CalculateEntropy(NewProbability)<=CalculateEntropy(OldProbability)+C_SameEntropyMargin:
            return True
        return False

    def ApplyMissingInputs(self,CurrentALO:ALO):
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return
        OldProbability=CurrentALO.GetJoinedProbability()
        NewProbability=0.5*(1-OldProbability)
        NewProbability+=OldProbability
        CurrentALO.SetJoinedProbability(NewProbability)

class XORNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.PrepareForNextPropagation()

    def __repr__(self):
        return f"XORNode:(name='{self._Name}')"

    def PrepareForNextPropagation(self):
        super().PrepareForNextPropagation()
        NewALO=ALO(1,C_NumberofTrialMax,AVec(),[])
        self._ALO=[NewALO]

    #
    #   Access internal data
    #

    def IfOperationNode(self) -> bool :
        return True

    def IfXORNode(self) -> bool :
        return True

    def CreateCloneNode(self) :
        NewNode=XORNode(self.getname()+"+")
        NewNode._CopySourceNode=self.GetCopySourceNode()
        return NewNode
    
    def GetNodeType(self) -> NodeType:
        return NodeType.XOR

    def SubstituteValue(self,P:float,N:float):
        CurrentALO=self.GetCurrentALO()
        if CurrentALO is None:
            return
        if CurrentALO._JoinedInputs==0:
            NewProbability=P
        else:
            # P'=P(1-2NewP)+NewP=P+NewP-2PNewP
            OldProbability=CurrentALO.GetJoinedProbability()
            P1=2*OldProbability*P
            NewProbability=P+OldProbability-P1
        CurrentALO.SetJoinedProbability(NewProbability)
        if (CurrentALO.GetJoinedN()>N):
            CurrentALO.SetJoinedN(N)
        CurrentALO._JoinedInputs+=1

    def GetOperationString(self) -> str:
        return " XOR "

    def AddActivation(self,CurrentALO:ALO,NewInputActivation):
        # Substitute to node inputs
        if (NewInputActivation==None):
            return
        if not self.CheckAVecRelation(CurrentALO.GetAVec(),NewInputActivation):
            return

        #if LogLevel>=2:
        #    Log("Lv2: Existing ALO for Node",self.getname())
        #    Log("Lv2: Existing ALO for Activation",NewInputActivation.getname())

        if len(CurrentALO._ParentActivations)==0:
            PropagatedProbability=NewInputActivation.GetPropagatedProbability()
            JN=NewInputActivation.GetPropagatedN()
            CurrentALO.SetJoinedProbability(PropagatedProbability)
            CurrentALO.SetJoinedN(JN)
            if LogLevel>=3:
                Log("Lv3: XOR operation : Joined ALO for Activation",self.getname())
                Log("   1 input JoinedProbability ",PropagatedProbability)
        else:
            # P'=P+NewP-2PNewP = 
            PropagatedProbability=NewInputActivation.GetPropagatedProbability()
            OldProbability=CurrentALO.GetJoinedProbability()
 
            P1=2*OldProbability*PropagatedProbability
            NewProbability=OldProbability+PropagatedProbability-P1
            CurrentALO.SetJoinedProbability(NewProbability)

            NewN=NewInputActivation.GetPropagatedN()
            if (CurrentALO.GetJoinedN()>NewN):
                CurrentALO.SetJoinedN(NewN)
            if LogLevel>=3:
                Log("Lv3: XOR operation : Joined ALO for Activation",self.getname())
                Log("   OldProbability",OldProbability,"ApplyingProbability",PropagatedProbability,"JoinedProbability",NewProbability)

        CurrentALO._ParentActivations.append(NewInputActivation)
        CurrentALO._AVec.JoinElements(NewInputActivation._AVec)
        ChangedRootActivation=NewInputActivation.GetChangedRootActivation()
        if ChangedRootActivation is not None and CurrentALO._ChangedRootActivation is None:
            CurrentALO._ChangedRootActivation=ChangedRootActivation
        CurrentALO._JoinedInputs+=1

    def JoinPWV(self,OldPWV:Propagation_weight_vector,NewPWV:Propagation_weight_vector) -> Propagation_weight_vector:
        if (OldPWV==None):
            OldPWV=NewPWV.CreateClone()
        else:
            OldPWV.XOROperation(NewPWV)
        return OldPWV
    
    def CheckPropagationReady(self,CurrentALO:ALO) -> bool:
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return True
        return False

    def ApplyMissingInputs(self,CurrentALO:ALO):
        if CurrentALO._JoinedInputs>=len(self._InputLinks):
            return
        
        # P'=P(1-2NewP)+NewP=P+NewP-2PNewP
        OldProbability=CurrentALO.GetJoinedProbability()
        P1=2*OldProbability*0.5
        NewProbability=0.5+OldProbability-P1
        CurrentALO.SetJoinedProbability(NewProbability)

class NodeVector:

    def __init__(self,NodeSize,name,node_class=RealNode):
        self._Nodes = []  # Nodesを初期化
        for i in range(NodeSize):
            eachname = name + str(i)
            self._Nodes.append(node_class(eachname))

    def Reset(self):
        for A in self._Nodes:
            A.Reset()

    def PrepareForNextPropagation(self):
        for A in self._Nodes:
            A.PrepareForNextPropagation()
            
    def ConnectbyMatrixto(self,AnotherNodeVector):
        for A in self._Nodes:
            AnotherNodeVector.ConnecttoAllFrom(A)
   
    def ConnecttoAllFrom(self,A):
        for B in self._Nodes:
            NewLink=g_NetworkFactory.CreateLogicLink(A,B,0.5,0.5,0,0,None)

    def SetIfObserved(self,Flag:bool):
        for A in self._Nodes:
            A.SetIfObserved(Flag)
    #
    #  Learning section 
    #
    def ActivateObservations(self,CurrentStateActivation,InputVector,IfPropagateOnlyChanged=False,ComplementaryRootActivation=None):
       CA:Activation=CurrentStateActivation
       for i, OutNode in enumerate(self._Nodes):
            Value=InputVector[i].item()
            ResultActivation=OutNode.ActivateObservation(Value,CurrentStateActivation,IfPropagateOnlyChanged,ComplementaryRootActivation)

    def PropagateActivations(self):
        for Node in self._Nodes:
            Node.PropagateActivation()

    def PrepareNextPropagation(self):
        for Node in self._Nodes:
            Node.PrepareNextPropagation()

    #
    # Real value section
    #
    def SubstituteValue(self,ValueVector,InputN):
       for i, OutNode in enumerate(self._Nodes):
           Value=ValueVector[i].item()
           OutNode.SubstituteValue(Value,InputN)

    def PropagateValue(self):
        for Node in self._Nodes:
            Node.PropagateValue()

    def GetVectorResult(self):
        ResultVector=[]
        for Node in self._Nodes:
            ResultVector.append(Node.GetJoinedProbability())
        return ResultVector

    def DumpEquation(self,Depth=0,IfAll=False,IfRoot=False):
        for Node in self._Nodes:
            Equation,SMin,SMax,NMin=Node.DumpEquationTree(Depth,IfAll,IfRoot)
            Name=Node.getname()
            Log(" ",Name,"=",Equation)
    
    def DumpEquationTree(self,Index:int,Depth=0,IfAll=False,IfRoot=False):
            Equation,SMin,SMax,NMin=self._Nodes[Index].DumpEquationTree(Depth,IfAll,IfRoot)
            return Equation

    def PrepareForTestPropagation(self):
        for N in self._Nodes:
            N.PrepareForTestPropagation()

##########################################################################
# Link
##########################################################################

class Link:
    def __init__(self,InputNode:Node,OutputNode:Node,InitialP11=0.5,InitialP00=0.5,N11=0,N00=0,OriginatedLink=None):
        self._InputNode:Node=InputNode
        self._OutputNode:Node=OutputNode
        assert(InputNode!=OutputNode)
        assert(isinstance(InitialP11, float) or isinstance(InitialP11, int))
        assert(isinstance(InitialP00, float) or isinstance(InitialP00, int))

        self._InitialP11:float=InitialP11
        self._InitialP00:float=InitialP00
        self._P11P:float=1
        self._P11N:float=0
        self._N11P:float=InitialP11*N11
        self._N11N:float=(1-InitialP11)*N11
        self._P00P:float=1        
        self._P00N:float=0
        self._N00P:float=InitialP00*N00
        self._N00N:float=(1-InitialP00)*N00
        self._NDiffP:float=0
        self._NDiffN:float=0

        self._PathFeedbackP1Original:int=0
        self._PathFeedbackP0Original:int=0

        self._PreviousActivation:Activation=None
        self._CurrentActivation:Activation=None
        self._ConditionList:ConditionList=None
        self._OriginatedLink:Link=OriginatedLink
        if OriginatedLink is not None and self._InputNode is not None:
            self._PathFeedbackP1Original=self._OriginatedLink._PathFeedbackP1
            self._PathFeedbackP0Original=self._OriginatedLink._PathFeedbackP0
        self._OriginatedConditionList=None

        self._PathFeedbackP1=0
        self._PathFeedbackN1=0
        self._PathFeedbackP0=0
        self._PathFeedbackN0=0
        self._IfCandidate=False
        self._IfPropagate=True
        self._IfCreateCondition=False
        self._IfRandom=False
        self._IfPropagationUpdated=True

        g_diagnostics._Links+=1

    def __repr__(self):
        return f"Link:('{repr(self._InputNode)}' -> '{repr(self._OutputNode)}')"

    def PrepareForNextPropagation(self):
        if self._CurrentActivation is not None:
            self._PreviousActivation=self._CurrentActivation
            self._PreviousActivation.PrepareForNextPropagation()

    def RemoveReferences(self):
        if self._OutputNode is not None:
            self._OutputNode.RemoveInputLink(self)
        if self._InputNode is not None:
            self._InputNode.RemoveOutputLink(self)

    def GetInitialLinkSign(self) -> bool:
        # Use initial Probability
        return (self._InitialP11+self._InitialP00) >= 0.5*2

    def GetLinkSign(self) -> bool:
        P1,N1=self.GetAverageP11N11()
        P0,N0=self.GetAverageP00N00()
        if N1==0 or N0==0:
            return self.GetInitialLinkSign()
        elif P1+P0 >= 0.5*2:
            return True
        else:
            return False
        
    def GetPreviousActivation(self) :
        return self._PreviousActivation
            
    def GetOrCreateConditionList(self) -> ConditionList:
        if self._ConditionList is None:
            self._ConditionList=ConditionList()
        return self._ConditionList
 
    def GetOriginLinkRecursively(self) -> "Link":
        if self._OriginatedLink is not None:
            return self._OriginatedLink.GetOriginLinkRecursively()
        else:
            return self
        
    def GetP11(self) -> float:
        return self._InitialP11
    
    def GetP00(self) -> float:
        return self._InitialP00

    def GetN11(self) -> float:
        return self._N11P+self._N11N
    
    def GetN00(self) -> float:
        return self._N00P+self._N00N

    def GetN11Feedbacked(self) -> float:
        # Feedbacked side
        if self._InitialP11>0.5:
            return self._N11N
        else:
            return self._N11P

    def GetN00Feedbacked(self) -> float:
        # Feedbacked side
        if self._InitialP00>0.5:
            return self._N00N
        else:
            return self._N00P

    def GetAverageP11N11(self) -> {float,float}:
        N=0
        P=0
        if self._N11P>0:
            P+=self._P11P*self._N11P
            N+=self._N11P
        if self._N11N>0:
            P+=self._P11N*self._N11N
            N+=self._N11N
        if N>0:
            P=P/N
            return P,N
        else:
            return self._InitialP11,N

    def GetAverageP00N00(self) -> {float,float}:
        N=0
        P=0
        if self._N00P>0:
            P+=self._P00P*self._N00P
            N+=self._N00P
        if self._N00N>0:
            P+=self._P00N*self._N00N
            N+=self._N00N
        if N>0:
            P=P/N
            return P,N
        else:
            return self._InitialP00,N
    
    def GetTotalEntropy(self) -> float:
        # NOT using
        P11,N11=self.GetAverageP11N11()
        P00,N00=self.GetAverageP00N00()
        S11=CalculateEntropy(P11)
        S00=CalculateEntropy(P00)
        return (S11+S00)/2

    def PropagateValue(self,P:float,N:float):
        P11,N11=self.GetAverageP11N11()
        P00,N00=self.GetAverageP00N00()
        PropagatedP=P*P11+((1-P)*(1-P00))
        NDash=P*N11+(1-P)*N00
        if (NDash > N):
            # Minimum N
            NDash=N
        assert(NDash>=0)
        self._OutputNode.SubstituteValue(PropagatedP,NDash)
    
    def CalculateREWeight(P:float,N:float,IfP11:bool) -> {float,float,float}:
        # for unit test only
        P1100=0
        Weight=0
        if (N<=0):
            P1100=P
            Weight=P1100*(1-P1100)
            Weight*=C_MaxWeightValue
        else:
            P1100=((P*N)+0.5)/(N+1)
            Weight=P1100*(1-P1100)
            Weight/=N+1

        if IfP11:
            R=P*P1100
            E=P
        else:
            R=(1-P1100)*(1-P)
            E=P-1       

        return R,E,Weight
    
    def CalculateWeight(self,LinkP11:float,LinkP00:float,LinkN11:float,LinkN00:float)->{float,float}:
       # Basically sign of weight value should be decided to decrease entropy of P11 or P00
        if (LinkN11<=0):
            P11=0.5
            P11Weight=P11*(1-P11)
            P11Weight*=C_MaxWeightValue
        else:
            P11=((LinkP11*LinkN11)+0.5)/(LinkN11+1)
            P11Weight=P11*(1-P11)
            P11Weight/=LinkN11+1

        if (LinkN00<=0):
            P00=0.5
            P00Weight=P00*(1-P00)
            P00Weight*=C_MaxWeightValue
        else:
            P00=((LinkP00*LinkN00)+0.5)/(LinkN00+1)
            P00Weight=P00*(1-P00)
            P00Weight/=LinkN00+1

        return P11Weight,P00Weight
    
    def IfReadyForDumpEquation(self):
        if self._N11N>0 or self._N00N>0:
            return False
        return True
        
    def ActivateLink(self,CurrentALO:ALO) -> "Activation":
        # Weight calcuration
        # Weight=P(1-P)/N
        P:float=CurrentALO.GetJoinedProbability()
        N:float=CurrentALO.GetJoinedN()
        AVec=CurrentALO.GetAVec()
        if AVec is None:
            assert(False)

        ParentActivations=CurrentALO.GetParentActivations()

        # Propagation
        # Use forward Probability
        P11=self.GetP11()
        N11=self.GetN11()
        P00=self.GetP00()
        N00=self.GetN00()
        PropagatedP=(P*P11)+((1-P)*(1-P00))
        NewN=(P*N11)+((1-P)*N00)
        PropagatedN=N
        if (PropagatedN>NewN):
            # Minimum N
            PropagatedN=NewN
        assert(PropagatedN>=0)    
        if LogLevel>=3:
            Log("Lv3 : Activate link ",self._InputNode._Name," to ",self._OutputNode._Name)
            Log("   P11,P00",P11,P00," N11,N00",N11,N00," PropagatedP,N",PropagatedP,PropagatedN)

        # Create new activation
        PreviousActivation=self.GetPreviousActivation()
        ChangedRootActivation=CurrentALO.GetChangedRootActivation()
        PropagatingActivation=Activation(P,PropagatedP,PropagatedN,self,None,None,ParentActivations,AVec,PreviousActivation,ChangedRootActivation)

        # Use finished activation
        self._OutputNode.ActivateNode(PropagatingActivation)
        self._CurrentActivation=PropagatingActivation

        return PropagatingActivation

    def ApplyFeedbackP11(self,P11Modified:float,N11Add:float):
        if P11Modified>=0.5:
            N11Updated=self._N11P+N11Add
            if (N11Updated>0):
                P11Updated=(self._P11P*self._N11P)+(P11Modified*N11Add)
                P11Updated/=N11Updated
                self._P11P=P11Updated
                self._N11P=N11Updated
        else:
            N11Updated=self._N11N+N11Add
            if (N11Updated>0):
                P11Updated=(self._P11N*self._N11N)+(P11Modified*N11Add)
                P11Updated/=N11Updated
                self._P11N=P11Updated
                self._N11N=N11Updated
        if self._InitialP11==0.5:
            if self._N11P>0 and self._N00P>0:
                self._InitialP11=1.0
                self._InitialP00=1.0
            elif self._N11N>0 and self._N00N>0:
                self._InitialP11=0.0
                self._InitialP00=0.0

    def ApplyFeedbackP00(self,P00Modified:float,N00Add:float):
        if P00Modified>=0.5:
            N00Updated=self._N00P+N00Add
            if (N00Updated>0):
                P00Updated=(self._P00P*self._N00P)+(P00Modified*N00Add)
                P00Updated/=N00Updated
                self._P00P=P00Updated
                self._N00P=N00Updated
        else:
            N00Updated=self._N00N+N00Add
            if (N00Updated>0):
                P00Updated=(self._P00N*self._N00N)+(P00Modified*N00Add)
                P00Updated/=N00Updated
                self._P00N=P00Updated
                self._N00N=N00Updated
        if self._InitialP00==0.5:
            if self._N11P>0 and self._N00P>0:
                self._InitialP11=1.0
                self._InitialP00=1.0
            elif self._N11N>0 and self._N00N>0:
                self._InitialP11=0.0
                self._InitialP00=0.0
    
    def DumpEquationTree(self,DumpDepth,IfAll,IfRoot) -> tuple[str,float,float,float] :
        C_DumpEquationDepthMax=16
        if (DumpDepth>C_DumpEquationDepthMax):
            return "",CalculateEntropy(1.0),CalculateEntropy(0.5),C_NumberofTrialMax
        LinkEq,SMin,SMax,ParentNMin=self._InputNode.DumpEquationTree(DumpDepth+1,IfAll,IfRoot)
        if LinkEq=="":
            return "",CalculateEntropy(1.0),CalculateEntropy(0.5),C_NumberofTrialMax
        P11,N11=self.GetAverageP11N11()
        P00,N00=self.GetAverageP00N00()
        S11=CalculateEntropy(P11)
        S00=CalculateEntropy(P00)
        if (SMin>S11):
            SMin=S11
        if (SMax<S11):
            SMax=S11
        if (SMin>S00):
            SMin=S00
        if (SMax<S00):
            SMax=S00
        C_DumpProbabilityThreshold0=0.01
        C_DumpProbabilityThreshold1=0.99
        if (P11>C_DumpProbabilityThreshold0 and P11<C_DumpProbabilityThreshold1) or (P00>C_DumpProbabilityThreshold0 and P00<C_DumpProbabilityThreshold1):
            LinkEq+="("+str(P11)+":"+str(P00)+")"
        if (P11<0.5) and (P00<0.5):
            FinalLinkEq="NOT "+LinkEq
        else:
            FinalLinkEq=LinkEq
        NMax=N11
        if NMax < N00:
            NMax = N00
        if ParentNMin> NMax:
            ParentNMin=NMax

        return FinalLinkEq,SMin,SMax,ParentNMin

class Activation:

    def __init__(self, SourceProbability:float,PropagatedProbability:float,N:float,PropagatingLink:Link,InputNode:Node,OutputNode:Node
                 ,ParentActivations,NewAVec:AVec,PreviousActivation=None,ComplementaryRootActivation=None):
        self._SourceProbability : float =SourceProbability
        self._PropagatedProbability : float = PropagatedProbability
        self._PropagatedN : float = N
        self._PropagatingLink : Link = PropagatingLink
        self._ParentActivations  = ParentActivations
        self._ChangedRootActivation = None
        
        PropagationDivisor=(2*SourceProbability)-1
        P11P00Value=(PropagatedProbability+SourceProbability-1)
        if PropagationDivisor!=0:
            P11P00Value/=PropagationDivisor
        else:
            P11P00Value=0.5
        self._CurrentP11 : float = P11P00Value
        self._CurrentP00 : float = P11P00Value
        self._CurrentN11 : float = 0
        self._CurrentN00 : float = 0
        if PropagatingLink is not None:
            self._CurrentP11 = PropagatingLink.GetP11()
            self._CurrentN11 = PropagatingLink.GetN11()
            self._CurrentP00 = PropagatingLink.GetP00()
            self._CurrentN00 = PropagatingLink.GetN00()
            self._InputNode = PropagatingLink._InputNode
            self._OutputNode = PropagatingLink._OutputNode
        else:
            self._InputNode : Node = InputNode
            self._OutputNode : Node = OutputNode
    
        assert(self._InputNode is not None)
        assert(self._OutputNode is not None)

        self._AVec: AVec = NewAVec
        assert(NewAVec!=None)
        self._IfP11Evaluated=False
        self._IfP00Evaluated=False

        if PropagatingLink is not None:
            PreviousActivation=self._PropagatingLink.GetPreviousActivation()
        if PreviousActivation is not None:
            self._PreviousSourceProbability : float = PreviousActivation._SourceProbability
            self._PreviousPropagatedProbability : float = PreviousActivation._PropagatedProbability
            self._PreviousP11 : float = PreviousActivation._CurrentP11
            self._PreviousP00 : float = PreviousActivation._CurrentP00
            self._IfPreviousProbabilityExists : bool = True
        else:
            self._PreviousSourceProbability : float = SourceProbability
            self._PreviousPropagatedProbability : float = PropagatedProbability
            self._PreviousP11 : float = 0
            self._PreviousP00 : float = 0
            self._IfPreviousProbabilityExists : bool = False
        self._ChangedRootActivation=ComplementaryRootActivation
        g_diagnostics._Activations+=1

    def getname(self) -> str:
        RS=""
        if self._InputNode is not None:
            RS+=self._InputNode.getname()
        RS+="->"
        if self._OutputNode is not None:
            RS+=self._OutputNode.getname()
        else:
            RS+="None"
        return RS
    
    def PrepareForNextPropagation(self):
        self._IfP11Evaluated = False
        self._IfP00Evaluated = False
        self._PreviousSourceProbability = self._SourceProbability
        self._PreviousPropagatedProbability = self._PropagatedProbability
        self._PreviousP11 = self._CurrentP11
        self._PreviousP00 = self._CurrentP00
        self._IfPreviousProbabilityExists = True

    def SetPreviousPropagation(self,SourceProbability:float,PropagatedProbability:float,P11:float,P00:float):
        self._PreviousSourceProbability = SourceProbability
        self._PreviousPropagatedProbability = PropagatedProbability
        self._PreviousP11 = P11
        self._PreviousP00 = P00
        self._IfPreviousProbabilityExists : bool = True
    
    def GetSourceProbability(self):
        return self._SourceProbability
        
    def GetPropagatedProbability(self):
        return self._PropagatedProbability

    def GetPropagatedN(self):
        return self._PropagatedN
        
    def GetAVec(self):
        return self._AVec
        
    def GetPropagatingLink(self):
        return self._PropagatingLink

    def GetChangedRootActivation(self):
        return self._ChangedRootActivation

    def GetCurrentP11(self) -> float:
        return self._CurrentP11
    
    def GetCurrentP00(self) -> float:
        return self._CurrentP00

    def GetPreviousP11(self) -> float:
        if self._IfPreviousProbabilityExists:
            return self._PreviousP11
        else:
            return 0.5

    def GetPreviousP00(self) -> float:
        if self._IfPreviousProbabilityExists:
            return self._PreviousP00
        else:
            return 0.5

    def GetSourceProbabilityFluctuation(self) -> float:
        if self._IfPreviousProbabilityExists:
            return self._SourceProbability-self._PreviousSourceProbability
        else:
            return 0

    def GetPropagatedProbabilityFluctuation(self) -> float:
        if self._IfPreviousProbabilityExists:
            return self._PropagatedProbability-self._PreviousPropagatedProbability
        else:
            return 0

    def GetDifferencialActivation(self):
        return self._ChangedRootActivation
    
    def PropagatePWVRecursively(self) -> Propagation_weight_vector:

        if self._ParentActivations==[]:
            JoinedPWV=Propagation_weight_vector(self._SourceProbability)
        else:
            JoinedPWV=None
            A:Activation
            for A in self._ParentActivations:
                ReturnPWV=A.PropagatePWVRecursively()
                JoinedPWV=self._InputNode.JoinPWV(JoinedPWV,ReturnPWV)

        P=self._SourceProbability
        PropagatingLink=self._PropagatingLink
        if PropagatingLink is not None:
            P11=PropagatingLink.GetP11()
            N11=PropagatingLink.GetN11()
            P00=PropagatingLink.GetP00()
            N00=PropagatingLink.GetN00()

            P11LinkWeight,P00LinkWeight=PropagatingLink.CalculateWeight(P11,P00,N11,N00)
            P11EntropyWeight=P11LinkWeight if P11>=0.5 else -P11LinkWeight
            P00EntropyWeight=P00LinkWeight if P00>=0.5 else -P00LinkWeight
            JoinedPWV.PropagateOnActivation(P11,P*P11EntropyWeight,P00,(1-P)*P00EntropyWeight)
            if P!=0:
                JoinedPWV.AddRootActivation(P,P11EntropyWeight,True,self)
            if 1-P!=0:
                JoinedPWV.AddRootActivation(-(1-P),P00EntropyWeight,False,self)
        else:
            JoinedPWV.PropagateOnActivation(self._CurrentP11,0,self._CurrentP00,0)
            
        return JoinedPWV

    def ApplyFeedbackP11(self, E: float, FeedbackaddValue: float, FeedbackN: float, ChangedRootActivation, PropagatedALO: ALO, IfCreateCondition: bool):
        """
        Apply feedback to the P11 probability value and update the propagated probability.
        Parameters:
        E (float): The effect value used in the feedback calculation.
        FeedbackaddValue (float): The value to be added to the current P11 value.
        FeedbackN (float): The feedback multiplier for the N11 value.
        ChangedRootActivation (Activation): 
        PropagatedALO (ALO): The propagated ALO object.
        IfCreateCondition (bool): Flag indicating whether to create a condition based on the feedback.
        Returns:
        None
        """
        # P11 side
        #Log(" Feedback P11 start Link",self._InputNode.getname(),"to", self._OutputNode.getname()," Feedback E,N",E,FeedbackN)
        P11=self._CurrentP11
        N11=self._CurrentN11
        N11Add=FeedbackN*abs(E)
        if (N11Add<=0):
            assert(N11Add==0)
            #Log("Feedback E*N==0")
            return
        P11Modified=P11+FeedbackaddValue
        # Saturation
        if (P11Modified>1):
            P11Modified=1
        if (P11Modified<0):
            P11Modified=0
        self._PropagatingLink.ApplyFeedbackP11(P11Modified,N11Add)
        self._CurrentP11=P11Modified
        IfFeedbacked=False
        if abs(P11Modified-self._PropagatingLink._InitialP11)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
            IfFeedbacked=True

        IfLinkPropagationFluctuated=False
        if self._IfPreviousProbabilityExists:
            if abs(self._CurrentP11-self._PreviousP11)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._SourceProbability-self._PreviousSourceProbability)<C_StableProbabilityThreshold:
                    # Source probability is stable
                    IfLinkPropagationFluctuated=True
            if abs(self._SourceProbability-self._PreviousSourceProbability)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._CurrentP11-self._PreviousP00)<C_StableProbabilityThreshold:
                    # Source probability fluctuated, and propagation probability is stable or reversed.
                    NDiffP=NDiffN=0
                    if self._CurrentP11>0.5 and self._PreviousP00>0.5:
                        if self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    if self._CurrentP11<0.5 and self._PreviousP00<0.5:
                        if not self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    self._PropagatingLink._NDiffP+=NDiffP
                    self._PropagatingLink._NDiffN+=NDiffN

        if (LogLevel>=2):
            # Updated propagated probability
            P=self._SourceProbability
            UpdatedPropagatedProbability=(P*self._CurrentP11)+((1-P)*(1-self._CurrentP00))
            Log("Lv2: Feedback P11 start Link",self._InputNode.getname(),"to", self._OutputNode.getname()," Feedback addP",FeedbackaddValue,"*E",E," N",FeedbackN)
            Log("  Feedback to P11 ",self.GetPreviousP11(), "to" ,P11Modified , " N11" , N11,"+" , N11Add)
            Log("  Updated probability  ",self._SourceProbability, "->",UpdatedPropagatedProbability)

        if IfCreateCondition and IfFeedbacked:
            if not self._IfP11Evaluated:
                g_SOLNetwork._CreateConditionFactory.EntryFromFeedback(self,ChangedRootActivation,PropagatedALO,True,IfLinkPropagationFluctuated,False)
                self._IfP11Evaluated=True

    def ApplyFeedbackP00(self,E:float,FeedbackaddValue:float,FeedbackN:float,ChangedRootActivation,PropagatedALO:ALO,IfCreateCondition:bool):
        # P00 side
        #Log(" Feedback P00 start link",self._InputNode.getname(),"to", self._OutputNode.getname()," Feedback E,N",E,FeedbackN)
        P00=self._CurrentP00
        N00=self._CurrentN00
        N00Add=FeedbackN*abs(E)
        if (N00Add<=0):
            assert(N00Add==0)
            #Log("Feedback E*N==0")
            return
        P00Modified=P00+FeedbackaddValue
        # Saturation
        if (P00Modified>1):
            P00Modified=1
        if (P00Modified<0):
            P00Modified=0
        self._PropagatingLink.ApplyFeedbackP00(P00Modified,N00Add)
        self._CurrentP00=P00Modified
        IfFeedbacked=False
        if abs(P00Modified-self._PropagatingLink._InitialP00)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
            IfFeedbacked=True
  
        IfLinkPropagationFluctuated=False
        if self._IfPreviousProbabilityExists:
            if abs(self._CurrentP00-self._PreviousP00)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._SourceProbability-self._PreviousSourceProbability)<C_StableProbabilityThreshold:
                    # Source probability is stable
                    IfLinkPropagationFluctuated=True
            if abs(self._SourceProbability-self._PreviousSourceProbability)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._CurrentP00-self._PreviousP11)<C_StableProbabilityThreshold:
                    # Source probability fluctuated, and propagation probability is stable or reversed.
                    NDiffP=NDiffN=0
                    if self._CurrentP00>0.5 and self._PreviousP11>0.5:
                        if self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    if self._CurrentP00<0.5 and self._PreviousP11<0.5:
                        if not self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    self._PropagatingLink._NDiffP+=NDiffP
                    self._PropagatingLink._NDiffN+=NDiffN

        if (LogLevel>=2):
            # Updated propagated probability
            P=self._SourceProbability
            UpdatedPropagatedProbability=(P*self._CurrentP11)+((1-P)*(1-self._CurrentP00))
            Log("Lv2: Feedback P00 start link",self._InputNode.getname(),"to", self._OutputNode.getname()," Feedback addP",FeedbackaddValue,"*E",E," N",FeedbackN)
            Log("  Feedback to P00 ",self.GetPreviousP00(), "to",P00Modified,  " N00" , N00,"+" , N00Add)
            Log("  Updated probability  ",self._SourceProbability, "->",UpdatedPropagatedProbability)
 
        if IfCreateCondition and IfFeedbacked:
            if not self._IfP00Evaluated:
                g_SOLNetwork._CreateConditionFactory.EntryFromFeedback(self,ChangedRootActivation,PropagatedALO,False,IfLinkPropagationFluctuated,False)
                self._IfP00Evaluated=True

    def FeedbackBalancedP11(self,E:float,FeedbackN:float,ChangedRootActivation,PropagatedALO:ALO):
        P11=self._PropagatingLink.GetP11()
        N11Add=FeedbackN*abs(E)
        if (N11Add<=0):
            return
        self._PropagatingLink.ApplyFeedbackP11(P11,N11Add)
        self._CurrentP11=self._PropagatingLink._InitialP11
  
        IfLinkPropagationFluctuated=False
        IfSourceProbabilityFluctuated=False
        if self._IfPreviousProbabilityExists:
            if abs(self._PreviousP11-self._PropagatingLink._InitialP11)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._SourceProbability-self._PreviousSourceProbability)<C_StableProbabilityThreshold:
                    # Source probability is stable
                    IfLinkPropagationFluctuated=True
            if abs(self._SourceProbability-self._PreviousSourceProbability)>0.5:
                if abs(self._CurrentP11-self._PreviousP00)<C_StableProbabilityThreshold:
                    IfSourceProbabilityFluctuated=True
                    NDiffP=NDiffN=0
                    if self._CurrentP11>0.5 and self._PreviousP00>0.5:
                        if self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    if self._CurrentP11<0.5 and self._PreviousP00<0.5:
                        if not self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    self._PropagatingLink._NDiffP+=NDiffP
                    self._PropagatingLink._NDiffN+=NDiffN

        if not self._IfP11Evaluated:
            if IfLinkPropagationFluctuated:
                g_SOLNetwork._CreateConditionFactory.EntryFromFeedback(self,ChangedRootActivation,PropagatedALO,True,IfLinkPropagationFluctuated,True)
            if IfSourceProbabilityFluctuated:
                g_SOLNetwork._AssociationFactory.EntryDifferencialActivation(self)
        self._IfP11Evaluated=True
    
    def FeedbackBalancedP00(self,E:float,FeedbackN:float,ChangedRootActivation,PropagatedALO:ALO):
        P00=self._PropagatingLink.GetP00()
        N00Add=FeedbackN*abs(E)
        if (N00Add<=0):
            return
        self._PropagatingLink.ApplyFeedbackP00(P00,N00Add)
        self._CurrentP00=self._PropagatingLink._InitialP00

        IfSourceProbabilityFluctuated=False
        IfLinkPropagationFluctuated=False
        if self._IfPreviousProbabilityExists:
            if abs(self._PreviousP00-self._PropagatingLink._InitialP00)>C_ProbabilityFluctuationToAddConditionIntoFeedbackedLink:
                if abs(self._SourceProbability-self._PreviousSourceProbability)<C_StableProbabilityThreshold:
                    # Source probability is stable
                    IfLinkPropagationFluctuated=True
            if abs(self._SourceProbability-self._PreviousSourceProbability)>0.5:
                if abs(self._CurrentP00-self._PreviousP11)<C_StableProbabilityThreshold:
                    IfSourceProbabilityFluctuated=True
                    NDiffP=NDiffN=0
                    if self._CurrentP00>0.5 and self._PreviousP11>0.5:
                        if self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    if self._CurrentP00<0.5 and self._PreviousP11<0.5:
                        if not self._PropagatingLink.GetInitialLinkSign():
                            NDiffP=1
                        else:
                            NDiffN=1
                    self._PropagatingLink._NDiffP+=NDiffP
                    self._PropagatingLink._NDiffN+=NDiffN

        if not self._IfP00Evaluated:
            if IfLinkPropagationFluctuated:
                g_SOLNetwork._CreateConditionFactory.EntryFromFeedback(self,ChangedRootActivation,PropagatedALO,False,IfLinkPropagationFluctuated,True)
            if IfSourceProbabilityFluctuated:
                g_SOLNetwork._AssociationFactory.EntryDifferencialActivation(self)
        self._IfP00Evaluated=True

    def ResumeFeedbackRecursively(self,RootActivation):
        for A in self._ParentActivations:
            A.ResumeFeedbackRecursively(RootActivation)
        if self!=RootActivation:
            if self._PropagatingLink is not None:
                self._PropagatingLink.ResumeFeedback()

    #
    # Copy network
    #
    def CopyAndInsertNewLinkFromTerminalRecursively(self,FBStartLink:Link,NewLogicNode:Node,LinkOutSign:bool) -> Link:
        # Copy links from FBStartLink to terminal activation recursively
        if FBStartLink._OutputNode==self._OutputNode:
            # Add to terminal
            LinkProbability=1 if LinkOutSign else 0
            NewLink=g_NetworkFactory.CreateLogicLink(NewLogicNode,self._OutputNode,LinkProbability,LinkProbability,0,0,FBStartLink)
            g_SOLNetwork.EntrytoMiddleLayer(self._OutputNode,NewLogicNode)
            return NewLink
        
        IfRootFound,FrontNodeBeforeTerminal,NewLogicLink=self.CreateLogicCopyFromTerminalRecursively(FBStartLink,NewLogicNode,LinkOutSign)
        if IfRootFound:
            TerminalNode=self._OutputNode
            TerminalLink=self.GetPropagatingLink()
            if TerminalLink is not None:
                TerminalLinkSign=TerminalLink.GetInitialLinkSign()
                TerminalProbability=1 if TerminalLinkSign else 0
                OriginatedLink=TerminalLink._OriginatedLink
                if OriginatedLink is None:
                    OriginatedLink=TerminalLink
                NewLink1=g_NetworkFactory.CreateLogicLink(FrontNodeBeforeTerminal,TerminalNode,TerminalProbability,TerminalProbability,0,0,OriginatedLink)
                #Log(": New Link (New logic node) ",ResultLogicNode.getname(),"->",TerminalNode.getname())
        # Return new link from the new logic node 
        return NewLogicLink
                
    def CopyLogicNodeSeparatingNewLogicNode(self,NewLogicNode:Node,A:"Activation",B:"Activation",ResultSign:bool) -> Link:
        CreatedNode=self._InputNode.CreateCloneNode()
        for PA in self._ParentActivations:
            if PA._InputNode==A._InputNode:
                continue
            if PA._InputNode==B._InputNode:
                continue
            CL=PA.GetPropagatingLink()
            if CL is None:
                continue
            NewInputNode=PA.CopyRootLinkRecursivelyFromSourceNode()
            g_NetworkFactory.CreateLogicLink(NewInputNode,CreatedNode,CL._InitialP11,CL._InitialP00,CL.GetN11(),CL.GetN00(),None)
        LinkProbability=1 if ResultSign else 0
        OriginatedLink=self._PropagatingLink
        return g_NetworkFactory.CreateLogicLink(NewLogicNode,CreatedNode,LinkProbability,LinkProbability,C_NumberofTrialMax,C_NumberofTrialMax,OriginatedLink)

    def CreateLogicCopyFromTerminalRecursively(self,FBLink:Link,NewLogicNode:Node,LinkOutSign:bool)-> {bool,Node,Link}:
        IfRootFound=False
        NewLogicLink=None
        if self._InputNode==FBLink._InputNode:
            # Copy starts from here
            g_SOLNetwork.EntrytoMiddleLayer(self._OutputNode,NewLogicNode)
            return True,NewLogicNode,NewLogicLink

        ParentCopyNode=None
        FoundParentActivation=None     
        A:Activation
        for A in self._ParentActivations:
            IfRootFound,ParentCopyNode,NewLogicLink=A.CreateLogicCopyFromTerminalRecursively(FBLink,NewLogicNode,LinkOutSign)
            if IfRootFound:
                FoundParentActivation=A
                break
        if not IfRootFound:
            return False,None,NewLogicLink
        
        # Copy input node and input links
        CreatedNode=self._InputNode.CreateCloneNode()
        g_SOLNetwork.EntrytoMiddleLayer(self._InputNode,CreatedNode)
        #Log(": CreateLogicCopyFromTerminalRecursively ",CreatedNode.getname())
 
        assert(ParentCopyNode is not None)
        IL:Link
        for IL in self._InputNode._InputLinks:
            NewLink:Link
            if IL._InputNode==FoundParentActivation._InputNode:
                if ParentCopyNode==NewLogicNode:
                    LinkProbability=1 if LinkOutSign else 0
                    NewLogicLink=g_NetworkFactory.CreateLogicLink(NewLogicNode,CreatedNode,LinkProbability,LinkProbability,0,0,FBLink)
                else:
                    NewLink=g_NetworkFactory.CreateLogicLink(ParentCopyNode,CreatedNode,IL._InitialP11,IL._InitialP00,IL.GetN11(),IL.GetN00(),None)
                #Log(": New Link ",ParentCopyNode.getname(),"->",CreatedNode.getname())
            else:
                NewLink=g_NetworkFactory.CreateLogicLink(IL._InputNode,CreatedNode,IL._InitialP11,IL._InitialP00,IL.GetN11(),IL.GetN00(),None)
                #Log(": New Link ",IL._InputNode.getname(),"->",CreatedNode.getname())
        # True : if copied path from RootActivation
        # CreatedNode : current result copy node 
        return True,CreatedNode,NewLogicLink

    def CopyRootLinkRecursivelyReplacingSourceNode(self,NewNode:Node) -> Node:
        # Copy parent nodes from this activation
        IfRootNode=True
        for PA in self._ParentActivations:
            if PA._PropagatingLink is not None:
                IfRootNode=False
        if IfRootNode:
            assert(False)
            return
        PA:Activation        
        for PA in self._ParentActivations:
            PreviousNode=PA.CopyRootLinkRecursivelyFromSourceNode()
            IL:Link
            IL=PA._PropagatingLink
            OriginatedLink=IL._OriginatedLink
            g_NetworkFactory.CreateLogicLink(PreviousNode,NewNode,IL._InitialP11,IL._InitialP00,IL.GetN11(),IL.GetN00(),None)
            #Log(" CopyRootLinkRecursivelyReplacingSourceNode",PreviousNode.getname(),"->",NewNode.getname())
        return

    def CopyRootLinkRecursivelyFromSourceNode(self) -> Node:
        IfRootNode=True
        for PA in self._ParentActivations:
            if PA._PropagatingLink is not None:
                IfRootNode=False
        if IfRootNode:
            return self._InputNode

        CopyNode=self._InputNode.CreateCloneNode()
        g_SOLNetwork.EntrytoMiddleLayer(self._InputNode,CopyNode)
        PA:Activation
        for PA in self._ParentActivations:
            IL:Link
            IL=PA._PropagatingLink
            if IL is not None:
                PreviousNode=PA.CopyRootLinkRecursivelyFromSourceNode()
                OriginatedLink=IL._OriginatedLink
                g_NetworkFactory.CreateLogicLink(PreviousNode,CopyNode,IL._InitialP11,IL._InitialP00,IL.GetN11(),IL.GetN00(),None)
                #Log(" CopyRootLink",PreviousNode.getname(),"->",CopyNode.getname())
        return CopyNode
    
    
##########################################################################
# Network manager
##########################################################################

class AssociationFactory:
    # 
    #  Create association with simultaneously fluctuated nodes
    #
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ObservationDifferencialMap = defaultdict(list)
            cls._instance._ActivationOnLinkDifferencialMap=defaultdict(list)

        return cls._instance
    
    def Reset(self):
        self._ObservationDifferencialMap.clear()
        self._ActivationOnLinkDifferencialMap.clear()

    def PrepareForNextPropagation(self):
        self._ObservationDifferencialMap.clear()
        self._ActivationOnLinkDifferencialMap.clear()

    def EntryDifferencialObservationALO(self,NewObservedALO:ALO):
        # Positive feedbacked
        DifferencialRootActivation=NewObservedALO._ChangedRootActivation
        self.CreateAssociationLinksFromALO(NewObservedALO)
        self._ObservationDifferencialMap[DifferencialRootActivation].append(NewObservedALO)

    def CreateAssociationLinksFromALO(self,NewObservedALO:ALO):
        # Create association link between simultaneously fluctuated nodes
        DifferencialRootActivation=NewObservedALO._ChangedRootActivation
        CALO:ALO
        NewAVec:AVec=NewObservedALO._AVec
        ALOList=self._ObservationDifferencialMap.get(DifferencialRootActivation)
        if ALOList is not None:
            for CALO in ALOList:
                if CALO is not NewObservedALO:
                    R=CALO._AVec.CompareSubstantialElements(NewAVec)
                    if R==SetCompareResult.Contain:
                        # Set of NewActivation is smaller
                        self.TryToCreateAssociationFromALOs(CALO,NewObservedALO)
                    elif R==SetCompareResult.Contained:
                        # Set of CA(CurrentActivation) is smaller
                        self.TryToCreateAssociationromALOs(NewObservedALO,CALO)
                
    def TryToCreateAssociationFromALOs(self,InputALO:ALO,OutputALO:ALO):
        InputNode=InputALO.GetCollidingNode()
        OutputNode=OutputALO.GetCollidingNode()
        if OutputNode._IfCandidateExists:
            return        
        for IL in OutputNode._InputLinks:
            if IL._InputNode==InputNode:
                # Already exists
                return

        PA=InputALO.GetJoinedProbability()
        PB=OutputALO.GetJoinedProbability()
        NewP11=PB*PA
        NewP00=(1-PB)*(1-PA)
        PropagationP=NewP11*PA+NewP00*(1-PA)
        NewLink=g_NetworkFactory.CreateLogicLink(InputNode,OutputNode,PropagationP,PropagationP,PA,1-PA,None)
        # Record first fluctuation
        NewLink._NDiffP=1
        # Activate on newly created link and entry
        NewDifferencialActivation=NewLink.ActivateLink(InputALO)
        # Set previous propagation
        PABar=1-PA
        PBBar=1-PB
        PrevP11=PABar*PBBar
        PrevP00=(1-PABar)*(1-PBBar)
        NewDifferencialActivation.SetPreviousPropagation(PABar,PBBar,PrevP11,PrevP00)
        self.EntryDifferencialActivation(NewDifferencialActivation)
        if (LogLevel>=1):
            Log("DEBUG : Lv2: Create association link",InputNode.getname(),"to", OutputNode.getname()," P11|P00",PropagationP)


    def EntryDifferencialActivation(self,NewActivation:Activation):
        if NewActivation._PropagatingLink is None:
            return
        if NewActivation._InputNode.IfRealNode() and NewActivation._OutputNode.IfRealNode():
            self._ActivationOnLinkDifferencialMap[NewActivation._OutputNode].append(NewActivation)

    def GetAssociatedActivations(self, TargetNode: Node) -> list[Link]:
        ActivationsList = self._ActivationOnLinkDifferencialMap.get(TargetNode)
        if ActivationsList is None:
            return None
        # _NDiffP+_NDiffN > 0 のリンクのみを抽出し、降順ソート
        return sorted(
            [A for A in ActivationsList if (A._PropagatingLink._NDiffP+A._PropagatingLink._NDiffN) > 0], 
            key=lambda A: (A._PropagatingLink._NDiffP+A._PropagatingLink._NDiffN), 
            reverse=True
            ) 

class FeedbackedObject:
    def __init__(self,FeedbackedActivation:Activation,ChangedRootActivation:Activation,TerminalALO:ALO,IfP11:bool,IfLinkPropagationFluctuated:bool
                 ,IfPreviousActivationFeedbacked:bool):
        self._FeedbackedActivation=FeedbackedActivation
        self._ChangedRootActivation=ChangedRootActivation
        self._TerminalALO=TerminalALO
        self._IfP11=IfP11
        self._IfLinkPropagationFluctuated=IfLinkPropagationFluctuated
        self._IfPreviousActivationFeedbacked=IfPreviousActivationFeedbacked

class CreateConditionFactory:
    #
    #  Create logic node
    #
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._FeedbackNodes = []  # インスタンス変数として初期化
            cls._instance._SerialNumber=1
        return cls._instance
    
    def Reset(self):
         self._FeedbackNodes.clear()

    def PrepareForNextPropagation(self):
         self._FeedbackNodes.clear()

    def CreateLink(self,Source:Node,Target:Node) -> Link:
        return Link(Source,Target)
    
    def EntryNewLink(self,NewLink :Link):
        pass
        
    def GetSerialNumber(self):
        NewNumberString=str(self._SerialNumber)
        self._SerialNumber+=1
        return NewNumberString
    
    def FindExistingLogicLink(self,FeedbackedLink:Link,CL:ConditionList,AL:ConditionList,ConditionNode:Node,ConditionSign:bool,LinkOutSign:bool,Type:NodeType) -> Link:
        # Search from the input node of the feedbacked link.
        NodeList=[]
        LinkSignList=[]
        CompareLogicNode=FeedbackedLink._InputNode
        if CompareLogicNode.GetNodeType()==Type:
            LI:Link
            for LI in CompareLogicNode._InputLinks:
                NodeList.append(LI._InputNode)
                LinkSignList.append(LI.GetLinkSign())
        else:
            NodeList.append(CompareLogicNode)
            LinkSignList.append(True)
        NodeList.append(ConditionNode)
        LinkSignList.append(ConditionSign)
        ResultLink=self.FindExistingLogicLinkFromLinkList(FeedbackedLink,CL,NodeList,LinkSignList,LinkOutSign,Type)
        if ResultLink is None and AL is not None:
            ResultLink=self.FindExistingLogicLinkFromLinkList(FeedbackedLink,AL,NodeList,LinkSignList,LinkOutSign,Type)
        return ResultLink

    def FindExistingLogicLinkXOR(self,FeedbackedLink:Link,InputLinkA:Link,InputLinkB:Link,CL:ConditionList) -> Link:
        NodeList=[]
        LinkSignList=[]
        OutputSign=FeedbackedLink.GetInitialLinkSign()
        SourceNodeA=InputLinkA._InputNode
        if SourceNodeA.IfXORNode():
            IL:Link
            for IL in SourceNodeA._InputLinks:
                NodeList.append(IL._InputNode)
                LinkSignList.append(True)
                OutputSign^=not IL.GetLinkSign()
        else:
            NodeList.append(SourceNodeA)
            LinkSignList.append(True)
            OutputSign^=not InputLinkA.GetLinkSign()

        SourceNodeB=InputLinkB._InputNode
        if SourceNodeB.IfXORNode():
            IL:Link
            for IL in SourceNodeB._InputLinks:
                NodeList.append(IL._InputNode)
                LinkSignList.append(True)
                OutputSign^=not IL.GetLinkSign()
        else:
            NodeList.append(SourceNodeB)
            LinkSignList.append(True)
            OutputSign^= not InputLinkB.GetLinkSign()

        return self.FindExistingLogicLinkFromLinkList(FeedbackedLink,CL,NodeList,LinkSignList,OutputSign,NodeType.XOR)
	
    def FindExistingLogicLinkFromLinkList(self,FeedbackedLink:Link,CL:ConditionList,NodeList,LinkSignList,LinkOutSign:bool,Type:NodeType) -> Link:
        # Search from the input node of the feedbacked link.
        if FeedbackedLink is None or CL is None:
            return None
        CLink:Link
        for CLink in CL._ConditionList:
            CInputNode=CLink._InputNode
            if CInputNode==FeedbackedLink._InputNode:
                continue
            if CInputNode.GetNodeType()==Type:
                IfAllMatched=self.SearchLogic_CheckLogicMatch(CLink,NodeList,LinkSignList,LinkOutSign)
                if IfAllMatched:
                    return CLink
        return None
    
    def SearchLogic_CheckLogicMatch(self,CandidateLink:Link,NodeList,LinkSignList,LinkOutSign:bool) -> bool:
        MatchedLinks=0
        IL:Link
        IL2:Link
        CandidateLogicNode=CandidateLink._InputNode
        for IL2 in CandidateLogicNode._InputLinks:
            for IN,ILS in zip(NodeList,LinkSignList):
                if IN==IL2._InputNode and (ILS==IL2.GetLinkSign()):
                    MatchedLinks+=1
                    break
                IL2Original=IL2._InputNode.GetCopySourceNode()
                if IN==IL2Original and (ILS==IL2.GetLinkSign()):
                    MatchedLinks+=1
                    break
        IfAllLinkMatched=(len(NodeList)==MatchedLinks) and (len(CandidateLogicNode._InputLinks)==MatchedLinks) and CandidateLink.GetLinkSign()==LinkOutSign
        return IfAllLinkMatched

#
#   Logic generator section
#
    def EntryFromFeedback(self,FBA:Activation,ChangedRootActivation:Activation,TerminalALO:ALO,IfP11:bool,IfLinkPropagationFluctuated:bool,IfPreviousActivationFeedbacked:bool):
        NewObject=FeedbackedObject(FBA,ChangedRootActivation,TerminalALO,IfP11,IfLinkPropagationFluctuated,IfPreviousActivationFeedbacked)
        self._FeedbackNodes.append(NewObject)
        g_diagnostics._Feedbacks+=1

    def CreateConditionNodes(self):
        RefObject:FeedbackedObject
        for RefObject in self._FeedbackNodes:
            FBA:Activation=RefObject._FeedbackedActivation
            FBLink=FBA.GetPropagatingLink()
            if FBLink is None or FBLink._InputNode is None or FBLink._OutputNode is None:
                continue
            # Check entropy reducing link
            TerminalActivation=RefObject._TerminalALO.GetTerminalActivation()
            IfCreateCondition=False
            FrontLink=None
            if TerminalActivation is not None:
                FrontLink=TerminalActivation.GetPropagatingLink()
                if FrontLink is not None:
                    IfCreateCondition=FrontLink._IfCreateCondition
            if RefObject._IfP11:
                if FBLink.GetP00()==0.5:
                    IfCreateCondition=False
            else:
                if FBLink.GetP11()==0.5:
                    IfCreateCondition=False
            ChangedRootActivation:Activation=RefObject._ChangedRootActivation
            TerminalALO:ALO=RefObject._TerminalALO
            IfP11:bool=RefObject._IfP11
            Result=True
            if not RefObject._IfPreviousActivationFeedbacked:
                Result=self.TryToCreateXORNodeFromExistingNode(FBA,TerminalALO,IfP11)
                if Result and IfCreateCondition:
                    if RefObject._IfLinkPropagationFluctuated:
                        self.CreateConditionNodeFluctuated(FBA,ChangedRootActivation,TerminalALO,IfP11,RefObject._IfPreviousActivationFeedbacked)
                    elif FrontLink is not None and FrontLink._OriginatedLink is not None:
                        self.CreateConditionNodeStatic(FBA,ChangedRootActivation,TerminalALO,IfP11)

    def TryToCreateXORNodeFromExistingNode(self,FeedbackedActivation:Activation,TerminalALO:ALO,IfP11:bool) -> bool:
        # Create XOR node from existing AND node or OR node
        A=FeedbackedActivation
        FeedbackNode:Node=FeedbackedActivation._InputNode
        if FeedbackNode.IfANDNode():
            # AND Output feedbacked and(0,0,0..) = 0->1
            if not IfP11:
                XORCandidateA=None
                XORCandidateB=None
                NumberofXORInputs=0
                for PA in FeedbackedActivation._ParentActivations:
                    LogicInputProbability=PA._PropagatedProbability
                    if (LogicInputProbability<0.5):
                        if XORCandidateA is None: 
                            XORCandidateA=PA
                        else:
                            XORCandidateB=PA
                        NumberofXORInputs+=1
                if NumberofXORInputs==2:
                    if self.CreateXORNodeFromFeedback(A,XORCandidateA,XORCandidateB,TerminalALO):
                        # XOR created
                        return False
                elif NumberofXORInputs==1:
                    # AND (0,1,1..) (1,1,0..)
                    # Skip creating condition. Condition should be inserted to the input link.
                    return True
            return True                
        elif FeedbackNode.IfORNode():
            # OR Output feedbacked or(1,1,1...) = 1->0
            if IfP11:
                XORCandidateA=None
                XORCandidateB=None
                NumberofXORInputs=0
                for PA in FeedbackedActivation._ParentActivations:
                    LogicInputProbability=PA._PropagatedProbability
                    if (LogicInputProbability>0.5):
                        if XORCandidateA is None: 
                            XORCandidateA=PA
                        else:
                            XORCandidateB=PA
                        NumberofXORInputs+=1
                if NumberofXORInputs==2:
                    if self.CreateXORNodeFromFeedback(A,XORCandidateA,XORCandidateB,TerminalALO):
                        # XOR created
                        return False
                elif NumberofXORInputs==1:
                    # OR (1,0,1..) (0,1,1..)
                    # Skip creating condition. Condition should be inserted to the input link.
                    return True
            return True                
        elif FeedbackNode.IfXORNode():
            return True
        else:
            return True
        # Result : True if continue creating conditon. 
    
    def CreateConditionNodeFluctuated(self,FeedbackedActivation:Activation,ChangedRootActivation:Activation,TerminalALO:ALO,IfP11:bool,IfPreviousActivationFeedbacked:bool):
        # Normal feedback section
        Result=False
        A=FeedbackedActivation
        CollidingAVec=TerminalALO._AVec

        SourceProbabilityFluctuationABS=abs(A.GetSourceProbabilityFluctuation())
        if SourceProbabilityFluctuationABS<C_StableProbabilityThreshold:
            # Association by simultaneously fluctuated
            # AND or OR logic creation
            if len(TerminalALO._ParentActivations)!=1:
                return
            TerminalNode=TerminalALO._ParentActivations[0]._OutputNode
            BList=g_SOLNetwork._AssociationFactory.GetAssociatedActivations(TerminalNode)
            if not BList:
                return
            CollidingNode:Node=TerminalALO.GetCollidingNode()

            B:Activation
            for B in BList:
                if not B._IfPreviousProbabilityExists:
                    continue
                if LogLevel>=3:
                    Log("Lv3: CreateConditionNode: Candidate ActivationA",A._InputNode.getname(),"->",A._OutputNode.getname()," ActivationB target",B._OutputNode.getname())
                # Check AVec
                BAVec=B.GetAVec()
                # Coompare AVec
                if BAVec is None:
                    if LogLevel>=4:
                        Log("Lv4: Propagation probabiity No Compare AVec")
                    continue
                CompareResult=CollidingAVec.CompareSubstantialElements(BAVec)
                if not CompareResult==SetCompareResult.Same and not CompareResult==SetCompareResult.Contain:
                    # B is smaller
                    continue
                CompareResult=CollidingAVec.CompareOuterElements(BAVec)
                if not CompareResult==SetCompareResult.NoRelation:
                    # Condition already included
                    continue
                else:
                    Result=False
                    if IfP11:
                        Result=self.CreateConditionLinkP11(A,B,TerminalALO,IfPreviousActivationFeedbacked)
                    else:
                        Result=self.CreateConditionLinkP00(A,B,TerminalALO,IfPreviousActivationFeedbacked)
    
    def CreateConditionNodeStatic(self,FeedbackedActivation:Activation,ChangedRootActivation:Activation,TerminalALO:ALO,IfP11:bool):
        # Static feedback
        Result=False
        A=FeedbackedActivation
        CollidingAVec=TerminalALO._AVec
        TerminalLink=TerminalALO.GetTerminalLink()
        CollidingNode:Node=TerminalALO.GetCollidingNode()

        # Static association by positive feedbacked activations
        CL:Link
        CandidateConditions=[]
        SearchingList=TerminalLink._OriginatedConditionList
        if SearchingList is not None:
            # Search from another logic nodes derived from the same link.
            CLink:Link
            for CLink in SearchingList._ConditionList:
                CN=CLink._InputNode
                if CN==A._InputNode or len(CN._OutputLinks)!=1:
                    continue
                CL=CN._OutputLinks[0]
                if not CL._IfPropagate:
                    continue
                CA=CL._CurrentActivation
                if CA is None:
                    continue
                FoundExisting=False
                if CN.IfANDNode() and IfP11 and CA.GetSourceProbability()<C_StableProbabilityThreshold:
                    #  AND(1,0,1...) = 0
                    PA:Activation
                    for PA in CA._ParentActivations:
                        if PA._InputNode.IfRealNode():
                            if PA.GetPropagatedProbability()<C_StableProbabilityThreshold:
                                A2:Activation
                                for A2 in CandidateConditions:
                                    if A2._InputNode==PA._InputNode:
                                        FoundExisting=True
                                if not FoundExisting:
                                    CandidateConditions.append(PA)
                if CN.IfORNode() and not IfP11 and CA.GetSourceProbability()>1-C_StableProbabilityThreshold:
                    #  OR(0,1,0...) = 1
                    PA:Activation
                    for PA in CA._ParentActivations:
                        if PA._InputNode.IfRealNode():
                            if PA.GetPropagatedProbability()>(1-C_StableProbabilityThreshold):
                                A2:Activation
                                for A2 in CandidateConditions:
                                    if A2._InputNode==PA._InputNode:
                                        FoundExisting=True
                                if not FoundExisting:
                                    CandidateConditions.append(PA)
        BA:Activation
        for BA in CandidateConditions:
            if LogLevel>=3:
                Log("Lv3: CreateConditionNode: Candidate ActivationA",A._InputNode.getname(),"->",A._OutputNode.getname(),
                    " ActivationB target",BA._OutputNode.getname())
            # Check AVec
            CompareAVec=BA.GetAVec()
            # Coompare AVec
            if CompareAVec is None:
                if LogLevel>=4:
                    Log("Lv4: Propagation probabiity No Compare AVec")
                continue
            CompareResult=CollidingAVec.CompareOuterElements(CompareAVec)
            if not CompareResult==SetCompareResult.NoRelation:
                # Condition already included
                continue
            else:
                Result=False
                if IfP11:
                    Result=self.CreateConditionLinkP11(A,BA,TerminalALO,False)
                else:
                    Result=self.CreateConditionLinkP00(A,BA,TerminalALO,False)

    def InsertNewLogicNode(self,NewLogic:Node,InsertingLink:Link,NewLinkSign:bool) -> Link:
        # NOT USING
        OutputNode=InsertingLink._OutputNode
        assert(OutputNode!=None)
        InsertingLink._OutputNode.RemoveInputLink(InsertingLink)
        InsertingLink._OutputNode=NewLogic
        NewLogic.EntryInputLink(InsertingLink)
        LinkSign=1 if NewLinkSign else 0
        NewLink=g_NetworkFactory.CreateLogicLink(NewLogic,OutputNode,LinkSign,LinkSign,0,0)
        return NewLink

    def CreateConditionLinkP11(self,A:Activation,B:Activation,TerminalALO:ALO,IfPreviousActivationFeedbacked:bool) -> bool:
        FeedbackedLink:Link=A._PropagatingLink
        AnotherLink:Link=B._PropagatingLink
        if FeedbackedLink is None:
            return False
        if AnotherLink is None:
            return False
        ConditionNode:Node=B._InputNode
        ConditionProbability:float=B._SourceProbability
        ConditionProbabilityPrevious:float=B._PreviousSourceProbability
        # Inserting may be modified by another CreateCondition
        LinkOutSign=True
        LinkP00A=FeedbackedLink._InitialP00
        if LinkP00A>0.5:
            LinkOutSign=True
        else:
            LinkOutSign=False
        if IfPreviousActivationFeedbacked:
            FeedbackedConditionProbability=ConditionProbabilityPrevious
        else:
            FeedbackedConditionProbability=ConditionProbability
        LinkBSign=FeedbackedConditionProbability<0.5 #  1 AND 0

        if LogLevel>=2:
            Log("Lv2: Create AND node")
            Log("   Feedbacked link P11  source",A._PreviousSourceProbability,"->",A._SourceProbability," P11 prev",A.GetPreviousP11(),"cur",A._CurrentP11,"P00",A._CurrentP00)
            Log("   Another link ",ConditionNode.getname()," Probability",ConditionProbability)

        Result=self.CreateConditionLinkP11Body(A,TerminalALO,FeedbackedLink,AnotherLink,LinkBSign,LinkOutSign)
        if not Result:
            # Reverse sign
            Result=self.CreateConditionLinkP11Body(A,TerminalALO,FeedbackedLink,AnotherLink,not LinkBSign,not LinkOutSign)
        return Result
    
    def CreateConditionLinkP11Body(self,A:Activation,TerminalALO:ALO,FeedbackedLink:Link,AnotherLink:Link,LinkBSign:bool,LinkOutSign:bool) -> bool:
        SourceNode=FeedbackedLink._InputNode
        TargetNode=FeedbackedLink._OutputNode
        TerminalActivation=TerminalALO.GetTerminalActivation()
        ConditionNode:Node=AnotherLink._InputNode

        if SourceNode.IfANDNode():
            CL:ConditionList=FeedbackedLink.GetOrCreateConditionList()
            AL:ConditionList=None
            if AnotherLink._OutputNode==FeedbackedLink._OutputNode:
                AL=AnotherLink.GetOrCreateConditionList()
            FoundLogicLink=self.FindExistingLogicLink(FeedbackedLink,CL,AL,ConditionNode,LinkBSign,LinkOutSign,NodeType.AND)
            if FoundLogicLink is not None:
                if LogLevel>=3:
                    FoundLogicNode=FoundLogicLink._InputNode
                    Log("Lv3: Use existing Logic node",FoundLogicNode.getname()," Connecting node",TargetNode.getname(),"=",SourceNode.getname(),"AND",ConditionNode.getname())
                return False
            # Add to existing logic node
            CloneSourceNode=SourceNode
            NewLogicNode=ANDNode(CloneSourceNode.getname()+"_"+self.GetSerialNumber())
            A.CopyRootLinkRecursivelyReplacingSourceNode(NewLogicNode)
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,NewLogicNode,LinkOutSign)
            LinkProbability=1 if LinkBSign else 0
            NewLink2=g_NetworkFactory.CreateLogicLink(ConditionNode,NewLogicNode,LinkProbability,LinkProbability,FeedbackedLink.GetN11(),0,FeedbackedLink)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(NewLogicLink)
            if AL is not None:
                AL._ConditionList.append(NewLogicLink)
            if LogLevel>=1:
                Log("Lv1: Add to existing AND node ",NewLogicNode.getname(),"=",CloneSourceNode.getname(),"AND",ConditionNode.getname() ,
                    " Link2",NewLink2.GetLinkSign(),"OutLink",LinkOutSign)
        elif TargetNode.IfANDNode():
            # Ignore feedback
            return True
        else:
            CL:ConditionList=FeedbackedLink.GetOrCreateConditionList()
            AL:ConditionList=None
            if AnotherLink._OutputNode==FeedbackedLink._OutputNode:
                AL=AnotherLink.GetOrCreateConditionList()
            FoundLogicLink=self.FindExistingLogicLink(FeedbackedLink,CL,AL,ConditionNode,LinkBSign,LinkOutSign,NodeType.AND)
            if FoundLogicLink is not None:
                if LogLevel>=3:
                    FoundLogicNode=FoundLogicLink._InputNode
                    Log("Lv3: Use existing Logic node",FoundLogicNode.getname()," Connecting node",TargetNode.getname(),"=",SourceNode.getname(),"AND",ConditionNode.getname())
                return False
            ConnectingNode=TargetNode
            NewLogicNode=ANDNode(TargetNode.getname()+"AND"+self.GetSerialNumber())
            SourceNodeClone=A.CopyRootLinkRecursivelyFromSourceNode()
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,NewLogicNode,LinkOutSign)
            NewLink=g_NetworkFactory.CreateLogicLink(SourceNodeClone,NewLogicNode,1,1,FeedbackedLink.GetN11(),0,FeedbackedLink)
            LinkProbability=1 if LinkBSign else 0
            NewLink2=g_NetworkFactory.CreateLogicLink(ConditionNode,NewLogicNode,LinkProbability,LinkProbability,FeedbackedLink.GetN11(),0,FeedbackedLink)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(NewLogicLink)
            if AL is not None:
                AL._ConditionList.append(NewLogicLink)
            if LogLevel>=1:
                Log("Lv1: New AND node ",NewLogicLink._OutputNode.getname(),"=",NewLogicNode.getname(),"=",SourceNode.getname(),"AND",ConditionNode.getname(),
                    " Probability Link1",NewLink.GetLinkSign()," Link2",NewLink2.GetLinkSign()," OutLink",LinkOutSign)
        # True : Condition link created or already existing
        return True

    def CreateConditionLinkP00(self,A:Activation,B:Activation,TerminalALO:ALO,IfPreviousActivationFeedbacked:bool) -> bool:
        FeedbackedLink:Link=A._PropagatingLink
        AnotherLink:Link=B._PropagatingLink
        if FeedbackedLink is None:
            return False
        if AnotherLink is None:
            return False
        ConditionNode:Node=B._InputNode
        ConditionProbability:float=B._SourceProbability
        ConditionProbabilityPrevious:float=B._PreviousSourceProbability
        # Inserting may be modified by another CreateCondition
        LinkOutSign=True
        LinkP11A=FeedbackedLink._InitialP11
        if LinkP11A>0.5:
            LinkOutSign=True
        else:
            LinkOutSign=False
        if IfPreviousActivationFeedbacked:
            FeedbackedConditionProbability=ConditionProbabilityPrevious
        else:
            FeedbackedConditionProbability=ConditionProbability
        LinkBSign=FeedbackedConditionProbability>0.5 #  0 OR 1 = 1

        if LogLevel>=2:
            Log("Lv2: Create OR node")
            Log("   Feedbacked link P00  source",A._PreviousSourceProbability,"->",A._SourceProbability," P00 prev",A.GetPreviousP00(),"cur",A._CurrentP00,"P11",A._CurrentP11)
            Log("   Another link ",ConditionNode.getname()," Probability",ConditionProbability)

        Result=self.CreateConditionLinkP00Body(A,TerminalALO,FeedbackedLink,AnotherLink,LinkBSign,LinkOutSign)
        if not Result:
            # Reverse sign
            Result=self.CreateConditionLinkP00Body(A,TerminalALO,FeedbackedLink,AnotherLink,not LinkBSign,not LinkOutSign)
        return Result
            
    def CreateConditionLinkP00Body(self,A:Activation,TerminalALO:ALO,FeedbackedLink:Link,AnotherLink:Link,LinkBSign:bool,LinkOutSign:bool) -> bool:
        SourceNode=FeedbackedLink._InputNode
        TargetNode=FeedbackedLink._OutputNode
        TerminalActivation=TerminalALO.GetTerminalActivation()
        ConditionNode:Node=AnotherLink._InputNode

        if SourceNode.IfORNode():
            CL:ConditionList=FeedbackedLink.GetOrCreateConditionList()
            AL:ConditionList=None
            if AnotherLink._OutputNode==FeedbackedLink._OutputNode:
                AL=AnotherLink.GetOrCreateConditionList()
            FoundLogicLink=self.FindExistingLogicLink(FeedbackedLink,CL,AL,ConditionNode,LinkBSign,LinkOutSign,NodeType.OR)
            if FoundLogicLink is not None:
                if LogLevel>=3:
                    FoundLogicNode=FoundLogicLink._InputNode
                    Log("Lv3: Use existing Logic node",FoundLogicNode.getname()," Connecting node",TargetNode.getname(),"=",SourceNode.getname(),"OR",ConditionNode.getname())
                return True            # Add to existing logic node
            CloneSourceNode=SourceNode
            NewLogicNode=ORNode(CloneSourceNode.getname()+"_"+self.GetSerialNumber())
            A.CopyRootLinkRecursivelyReplacingSourceNode(NewLogicNode)
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,NewLogicNode,LinkOutSign)
            LinkProbability=1 if LinkBSign else 0
            NewLink2=g_NetworkFactory.CreateLogicLink(ConditionNode,NewLogicNode,LinkProbability,LinkProbability,0,FeedbackedLink.GetN00(),FeedbackedLink)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(NewLogicLink)
            if AL is not None:
                AL._ConditionList.append(NewLogicLink)
            if LogLevel>=1:
                Log("Lv1: Add to existing OR node ",NewLogicNode.getname(),"=",SourceNode.getname(),"OR",ConditionNode.getname() ,
                    " Link2",NewLink2.GetLinkSign(),"OutLink",LinkOutSign)
        elif TargetNode.IfORNode():
            # Ignore feedback
            return False
        else:
            CL:ConditionList=FeedbackedLink.GetOrCreateConditionList()
            AL:ConditionList=None
            if AnotherLink._OutputNode==FeedbackedLink._OutputNode:
                AL=AnotherLink.GetOrCreateConditionList()
            FoundLogicLink=self.FindExistingLogicLink(FeedbackedLink,CL,AL,ConditionNode,LinkBSign,LinkOutSign,NodeType.OR)
            if FoundLogicLink is not None:
                if LogLevel>=3:
                    FoundLogicNode=FoundLogicLink._InputNode
                    Log("Lv3: Use existing Logic node",FoundLogicNode.getname()," Connecting node",TargetNode.getname(),"=",SourceNode.getname(),"AND",ConditionNode.getname())
                return True
            ConnectingNode=TargetNode
            NewLogicNode=ORNode(TargetNode.getname()+"OR"+self.GetSerialNumber())
            SourceNodeClone=A.CopyRootLinkRecursivelyFromSourceNode()
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,NewLogicNode,LinkOutSign)
            NewLink=g_NetworkFactory.CreateLogicLink(SourceNodeClone,NewLogicNode,1,1,0,FeedbackedLink.GetN00(),FeedbackedLink)
            LinkProbability=1 if LinkBSign else 0
            NewLink2=g_NetworkFactory.CreateLogicLink(ConditionNode,NewLogicNode,LinkProbability,LinkProbability,0,FeedbackedLink.GetN00(),FeedbackedLink)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(NewLogicLink)
            if AL is not None:
                AL._ConditionList.append(NewLogicLink)
            if LogLevel>=1:
                Log("Lv1: New OR node ",NewLogicLink._OutputNode.getname(),"=",NewLogicNode.getname(),"=",SourceNode.getname(),"OR",ConditionNode.getname() ,
                    " Probability Link1",NewLink.GetLinkSign()," Link2",NewLink2.GetLinkSign()," OutLink",LinkOutSign)
        # True : Condition link created or already existing
        return True

    def CreateXORNodeFromFeedback(self,FeedbackedActivation:Activation,InputA1:Activation,InputA2:Activation,TerminalALO:ALO) -> bool:
    
        TargetLogicNode=FeedbackedActivation._InputNode
        #if len(FeedbackedActivation._ParentActivations)!=2:
            # Use only 2 input logic
        #    return False
        ExistingOutLink:Link=FeedbackedActivation.GetPropagatingLink()
        if ExistingOutLink is None:
            return False
        
        # Check input links
        InputL1=InputA1.GetPropagatingLink()
        InputL2=InputA2.GetPropagatingLink()
        if InputL1 is None or InputL2 is None:
            return False

        # Compare number of differencial probabilities
        OriginatedLink=ExistingOutLink._OriginatedLink
        if OriginatedLink is None:
            return False
        if OriginatedLink._NDiffP==0 or OriginatedLink._NDiffN==0:
            # 0->1 or 1->0 probability fluctutation should exist for creating XOR
            return False

        TargetLogicNode=FeedbackedActivation._InputNode
        Result=False
        if LogLevel>=2:
            Log("Lv2: CreateXORNodeFromFeedback candidate Node:",TargetLogicNode.getname(),"=",InputA1._InputNode.getname(),"XOR",InputA2._InputNode.getname())
        if TargetLogicNode.IfORNode() :
            OutputLinkSign=ExistingOutLink.GetLinkSign()
            if LogLevel>=3:
                NewLinkASign=InputA1._PropagatingLink.GetLinkSign()
                NewLinkBSign=InputA2._PropagatingLink.GetLinkSign()
                Log("Lv3: Try to create (from OR) XOR A,B LinkASign",NewLinkASign," LinkBSign",NewLinkBSign,"to " ,OutputLinkSign)
            Result=self.CreateXORNodeandLink(FeedbackedActivation,InputA1,InputA2,OutputLinkSign,TerminalALO)
        elif TargetLogicNode.IfANDNode():
            OutputLinkSign=not ExistingOutLink.GetLinkSign()
            if LogLevel>=3:
                NewLinkASign=InputA1._PropagatingLink.GetLinkSign()
                NewLinkBSign=InputA2._PropagatingLink.GetLinkSign()
                Log("Lv3: Try to create (from AND) NOT XOR A,B LinkASign",NewLinkASign," LinkBSign",NewLinkBSign,"to " ,OutputLinkSign)
            Result=self.CreateXORNodeandLink(FeedbackedActivation,InputA1,InputA2,OutputLinkSign,TerminalALO)

        return Result

    def CreateXORNodeandLink(self,FeedbackedActivation:Activation,A:Activation,B:Activation,OutputSign:bool,TerminalALO:ALO) -> bool:
        #Log(" Create XOR link source",A.GetSourceProbability()," propagated ",A._PreviousPropagatedProbability,"->",A._PropagatedProbability)
        
        FeedbackedLink=FeedbackedActivation._PropagatingLink
        TargetNode=FeedbackedActivation._OutputNode
        if FeedbackedLink is None:
            return False
        ConnectingNode:Node=FeedbackedLink._OutputNode
        if len(TerminalALO._ParentActivations)<1:
            return False
        LinkA=A._PropagatingLink
        LinkB=B._PropagatingLink
        LinkASign=LinkA.GetLinkSign()
        LinkBSign=LinkB.GetLinkSign()
        assert(A._OutputNode==FeedbackedLink._InputNode)
        assert(B._OutputNode==FeedbackedLink._InputNode)
        # Link propagation
        TerminalActivation=TerminalALO.GetTerminalActivation()
        CL:ConditionList=FeedbackedLink.GetOrCreateConditionList()
        FoundLogicLink=self.FindExistingLogicLinkXOR(FeedbackedLink,LinkA,LinkB,CL)
        if FoundLogicLink is not None:
            if LogLevel>=3:
                FoundLogicNode=FoundLogicLink._InputNode
                Log("Lv3: Use existing Logic node",FoundLogicNode.getname()," Connecting node",TargetNode.getname(),"=",A._InputNode.getname(),"XOR",B._InputNode.getname())
            return False        

        if LogLevel>=2:
            Log("Lv2: Create XOR node")
            Log("   Feedbacked link P00  source",A._PreviousSourceProbability,"->",A._SourceProbability," P00 prev",A.GetPreviousP00(),"cur",A._CurrentP00,"P11",A._CurrentP11)
            Log("   Another link  source",B._PreviousPropagatedProbability," to",B._PropagatedProbability," difference ",B.GetPropagatedProbabilityFluctuation())

        NewLogicNode=XORNode(ConnectingNode.getname()+"XOR"+self.GetSerialNumber())
        ResultSign=OutputSign
        if A._InputNode.IfXORNode() and len(A._ParentActivations)>0:
            for X in A._ParentActivations:
                LocalInputSign=X._PropagatingLink.GetLinkSign()
                ResultSign^=not LocalInputSign
                LocalInputLinkProbability=1
                SourceNodeClone=X.CopyRootLinkRecursivelyFromSourceNode()
                NewLink1=g_NetworkFactory.CreateLogicLink(SourceNodeClone,NewLogicNode,LocalInputLinkProbability,LocalInputLinkProbability,0,0,FeedbackedLink)
            ResultSign^=not LinkASign
        else:
            SourceNodeClone=A.CopyRootLinkRecursivelyFromSourceNode()
            ResultSign^=not LinkASign
            InputAProbability=1
            NewLink1=g_NetworkFactory.CreateLogicLink(SourceNodeClone,NewLogicNode,InputAProbability,InputAProbability,0,0,FeedbackedLink)
        if B._InputNode.IfXORNode() and len(B._ParentActivations)>0:
            for X in B._ParentActivations:
                LocalInputSign=X._PropagatingLink.GetLinkSign()
                ResultSign^=not LocalInputSign
                LocalInputLinkProbability=1
                SourceNodeClone=X.CopyRootLinkRecursivelyFromSourceNode()
                NewLink2=g_NetworkFactory.CreateLogicLink(SourceNodeClone,NewLogicNode,LocalInputLinkProbability,LocalInputLinkProbability,0,0,FeedbackedLink)
            ResultSign^=not LinkBSign
        else:
            ConditionNodeClone=B.CopyRootLinkRecursivelyFromSourceNode()
            ResultSign^=not LinkBSign
            InputBProbability=1
            NewLink2=g_NetworkFactory.CreateLogicLink(ConditionNodeClone,NewLogicNode,InputBProbability,InputBProbability,0,0,FeedbackedLink)

        if len(FeedbackedActivation._ParentActivations)>2:
            # Copy another OR|AND inputs.
            ModifiedLogicLink=FeedbackedActivation.CopyLogicNodeSeparatingNewLogicNode(NewLogicNode,A,B,ResultSign)
            # New XOR node and modified node from original >2 input AND|OR node
            ConnectingNode=ModifiedLogicLink._OutputNode
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,ConnectingNode,True)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            g_SOLNetwork.EntrytoMiddleLayer(ConnectingNode,NewLogicNode)

            # Insert new connecting logic node before OutputNode
            ModifiedLogicLink._OriginatedConditionList=CL
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(ModifiedLogicLink)
        else:
            # Replace from OR|AND node into XOR node
            NewLogicLink=TerminalActivation.CopyAndInsertNewLinkFromTerminalRecursively(FeedbackedLink,NewLogicNode,ResultSign)
            assert(NewLogicLink is not None)
            assert(CL is not None)
            NewLogicLink._OriginatedConditionList=CL
            CL._ConditionList.append(NewLogicLink)

        if LogLevel>=1:
            NewXORString=""
            if ResultSign:
                NewXORString="= XOR("
            else:
                NewXORString="= not XOR("
            IfFirst=True
            for NL in NewLogicNode._InputLinks:
                if IfFirst:
                    IfFirst=False
                else:
                    NewXORString+=","
                NewXORString+=NL._InputNode.getname()
            NewXORString+=")"
            Log("Lv1: New XOR node ",ConnectingNode.getname(),"=",NewLogicNode.getname(),NewXORString)
            Log(" original AND|OR node",NewLogicLink._OutputNode.getname(),"=",A._InputNode.getname(),"AND|OR",B._InputNode.getname() ," probability link1",NewLink1.GetLinkSign()," link2",NewLink2.GetLinkSign()," output link",ResultSign)
        return True
    
class SOLNetwork:

    def __init__(self,NumberOfInputs,NumberOfOutputs):

        self._GlobalRootNode=RealNode("GlobalRoot")
        self._InputVector=NodeVector(NumberOfInputs,"Input")
        self._OutputVector=NodeVector(NumberOfOutputs,"Output")
        self._InputVector.SetIfObserved(True)
        self._OutputVector.SetIfObserved(True)
        self._MiddleLayerNodes=[]
        self._CreateConditionFactory=CreateConditionFactory()
        self._AssociationFactory=AssociationFactory()

        # Root nodes
        self._RootN=C_NumberofTrialMax
        self._RootNode=RealNode("RootNode")
        self._TimeOriginNode=RealNode("TimeOriginNode")
        self._RootCausalityNode=RealNode("RootCausalityNode")
        self._RootCausalityActivation=self._RootCausalityNode.ActivateRoot(self._GlobalRootNode,1,self._RootN)

    def Reset(self):
        self._InputVector.Reset()
        self._OutputVector.Reset()
        for N in self._MiddleLayerNodes:
            N.Reset()
        self._CreateConditionFactory.Reset()
        self._AssociationFactory.Reset()

    def EntrytoMiddleLayer(self,ExistingNextNode,NewNode):
        try:
            # ExistingNodeの前に挿入
            index = self._MiddleLayerNodes.index(ExistingNextNode)
            self._MiddleLayerNodes.insert(index, NewNode)
        except ValueError:
            assert(not ExistingNextNode.IfOperationNode())
            # ExistingNodeがリスト内に見つからない場合
            self._MiddleLayerNodes.append(NewNode)


    def Learning(self,InputVector,ReferenceVector,CurrentID):

        self._InputVector.PrepareForNextPropagation()
        self._OutputVector.PrepareForNextPropagation()
        self._CreateConditionFactory.PrepareForNextPropagation()
        self._AssociationFactory.PrepareForNextPropagation()
        for N in self._MiddleLayerNodes:
            N.PrepareForNextPropagation()

        TimeNode=SeriesNode(self._TimeOriginNode,CurrentID)
        TimeActivation=TimeNode.ActivateRoot(self._GlobalRootNode,1,self._RootN)
        TimeActivationComplementary=TimeNode.ActivateRoot(self._GlobalRootNode,0,self._RootN)

        ParentInputActivation=self._RootNode.ActivateObservation(1.0,TimeActivation,False,False)
        ParentOutputActivation=self._RootNode.ActivateObservation(1.0,TimeActivation,False,False)
        # Add causality activation. Set of output observations are smaller. So,output nodes are connected as link output.
        ParentOutputActivation._AVec.AddSubstantialElement(self._RootCausalityActivation)

        # Condition nodes are input nodes. Output nodes should not be used as condition logic inputs.
        self._InputVector.ActivateObservations(ParentInputActivation,InputVector,True,TimeActivationComplementary)
        self._OutputVector.ActivateObservations(ParentOutputActivation,ReferenceVector,True,TimeActivationComplementary)

        # Start propagation
        self._InputVector.PropagateActivations()

        # Propagate on middle layers

#            CopyOfMiddleLayerNodes=[]         
        for N in self._MiddleLayerNodes:
#                CopyOfMiddleLayerNodes.append(N)
#            for N in CopyOfMiddleLayerNodes:
            N.PropagateActivation()
            # New network nodes will not be activated until the next iteration.

        self._OutputVector.PropagateActivations()

        # Compare ResultVector with OutputVector -> Apply feedback
        # Get output layer
        ResultVector=self._OutputVector.GetVectorResult()
        if LogLevel>=1:
            Log("Lv1: Learning  Result : InputVector ",InputVector," ResultVector ",ResultVector, " ReferenceVector ",ReferenceVector)

        ## Create condition logic using feedback result
        self._CreateConditionFactory.CreateConditionNodes()

        ## Finish feedbacked network
        self._OutputVector.PrepareNextPropagation()

    def GetResult(self,InputVector):
        self._InputVector.SubstituteValue(InputVector,C_NumberofTrialMax)
        self._InputVector.PropagateValue()
        for N in self._MiddleLayerNodes:
            N.PropagateValue()
        OutputVector=self._OutputVector.GetVectorResult()
        return OutputVector

    def DumpEquation(self,IfAll=False):
        self._OutputVector.DumpEquation(0,IfAll,True)

    @staticmethod
    def GenerateMatrix(dims, steps, random_ratio=0.1, priority_ratio=0.6):
        """
        冒頭4要素を優先的にランダム変動させ、それ以外の要素の変動を最小限にするベクトル列を生成。

        Args:
            dims (int): ベクトルの次元数。
            steps (int): 生成するステップ数。
            random_ratio (float): 完全ランダムにする確率。
            priority_ratio (float): 先頭4要素を変更する確率。

        Returns:
            list: 生成されたベクトルのリスト。
        """
        current_vector = torch.zeros(dims, dtype=torch.int32)
        sequence = [current_vector.tolist()]

        priority_indices = list(range(min(4, dims)))  # 先頭4要素のインデックス
        other_indices = list(range(4, dims))  # それ以外のインデックス

        for _ in range(steps - 1):
#            if g_random.random() < random_ratio:
#                new_vector = torch.randint(0, 2, (dims,), dtype=torch.int32)  # 完全ランダム
#            else:
            if True:
                new_vector = current_vector.clone()
                if g_random.random() < priority_ratio or not other_indices:
                    # 先頭4要素のどれかを変更
                    idx = g_random.choice(priority_indices)
                else:
                    # それ以外の要素を低頻度で変更
                    idx = g_random.choice(other_indices)

                new_vector[idx] = 1 - new_vector[idx]  # 反転
            
            sequence.append(new_vector.tolist())
            current_vector = new_vector

        return sequence


##########################################################################
# Unit tests
##########################################################################
    def Test_CreateTestLogic(self):
        # for TEST fixed
        # すべての0と1の組み合わせを生成
        XSize=4
        X = list(itertools.product([0, 1], repeat=XSize))

        # データセットの定義
        self._TestX = torch.tensor(X, dtype=torch.float32)
        Log('Inputs: ')
        Log(self._TestX)

        # 入力データを整数型に変換
        X_int = self._TestX.int()

        # 論理演算
        Y0 = X_int[:, 0] & X_int[:, 1]
        Y1 = X_int[:, 1] ^ X_int[:, 3]
        Y2 = (X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3])
        Y3 = (X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3])
        ReferenceEq=[
        "Y0 = X_int[:, 0] & X_int[:, 1]",
        "Y1 = X_int[:, 1] ^ X_int[:, 3]",
        "Y2 = (X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3])",
        "Y3 = X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3])",
        ]
        # 結果をfloat32に戻す
        Y0 = Y0.float()
        Y1 = Y1.float()
        Y2 = Y2.float()
        Y3 = Y3.float()

        # Y1とY2を結合して新しいテンソルYを作成
        # 
        #YVector=(Y0,Y1)
        YVector=(Y0,Y1,Y2,Y3)
        self._TestY = torch.stack(YVector, dim=1)
        self._TestYSize=len(YVector)

        #
        NumberOfInputs=4
        NumberOfOutputs=4
        self._TestIV=NodeVector(NumberOfInputs,"Input")
        self._TestOV=NodeVector(NumberOfOutputs,"Output")
        TestIVNodes=self._TestIV._Nodes
        TestOVNodes=self._TestOV._Nodes
        N=C_NumberofTrialMax
        Name0=TestOVNodes[0].getname()
        OutputNode0=g_NetworkFactory.CreateANDNode(Name0,TestIVNodes[0],TestIVNodes[1],True,True,N,N)
        NewLink0=g_NetworkFactory.CreateLogicLink(OutputNode0,TestOVNodes[0],1,1,N,N)
        self.EntrytoMiddleLayer(TestOVNodes[0],OutputNode0)
        Name1=TestOVNodes[1].getname()
        OutputNode1=g_NetworkFactory.CreateXORNode(Name1,TestIVNodes[1],TestIVNodes[3],True,True,N,N)
        NewLink1=g_NetworkFactory.CreateLogicLink(OutputNode1,TestOVNodes[1],1,1,N,N)
        self.EntrytoMiddleLayer(TestOVNodes[1],OutputNode1)
        Name2_1=TestOVNodes[2].getname()+"_1"
        OutputNode2_0=g_NetworkFactory.CreateORNode(Name2_1,TestIVNodes[0],TestIVNodes[1],True,True,N,N)
        Name2_2=TestOVNodes[2].getname()+"_2"
        OutputNode2_1=g_NetworkFactory.CreateORNode(Name2_2,TestIVNodes[2],TestIVNodes[3],True,True,N,N)
        Name2_3=TestOVNodes[2].getname()
        OutputNode2_2=g_NetworkFactory.CreateANDNode(Name2_3,OutputNode2_0,OutputNode2_1,True,True,N,N)
        NewLink2=g_NetworkFactory.CreateLogicLink(OutputNode2_2,TestOVNodes[2],1,1,N,N)
        self.EntrytoMiddleLayer(TestOVNodes[2],OutputNode2_0)
        self.EntrytoMiddleLayer(TestOVNodes[2],OutputNode2_1)
        self.EntrytoMiddleLayer(TestOVNodes[2],OutputNode2_2)
        Name3_1=TestOVNodes[3].getname()+"_1"
        OutputNode3_0=g_NetworkFactory.CreateXORNode(Name3_1,TestIVNodes[0],TestIVNodes[1],True,True,N,N)
        Name3_2=TestOVNodes[3].getname()+"_2"
        OutputNode3_1=g_NetworkFactory.CreateXORNode(Name3_2,TestIVNodes[2],TestIVNodes[3],True,True,N,N)
        Name3_3=TestOVNodes[3].getname()
        OutputNode3_2=g_NetworkFactory.CreateORNode(Name3_3,OutputNode3_0,OutputNode3_1,True,True,N,N)
        NewLink3=g_NetworkFactory.CreateLogicLink(OutputNode3_2,TestOVNodes[3],1,1,N,N)
        self.EntrytoMiddleLayer(TestOVNodes[3],OutputNode3_0)
        self.EntrytoMiddleLayer(TestOVNodes[3],OutputNode3_1)
        self.EntrytoMiddleLayer(TestOVNodes[3],OutputNode3_2)

#        Y0 = X_int[:, 0] & X_int[:, 1]
#        Y1 = X_int[:, 1] ^ X_int[:, 3]
#        Y2 = (X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3])
#        Y3 = (X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3])

    def Test_ValuePropagationTest(self,InputVector,ReferenceValue,CurrentID) -> bool :

        self._TestIV.Reset()
        self._OutputVector.Reset()
        self._TestOV.Reset()
        for N in self._MiddleLayerNodes:
            N.Reset()

        self._TestIV.PrepareForTestPropagation()
        for N in self._MiddleLayerNodes:
            N.PrepareForTestPropagation()

        self._TestIV.SubstituteValue(InputVector,C_NumberofTrialMax)
        self._TestIV.PropagateValue()
        for N in self._MiddleLayerNodes:
            N.PropagateValue()
        ResultVector=self._TestOV.GetVectorResult()

        Log("Predicted" , ResultVector)
        Log("Reference " ,ReferenceValue)
        IfMatched=True
        for j in range(len(ResultVector)):
            if (ResultVector[j]!=ReferenceValue[j]):
                IfMatched=False
        if (IfMatched):
            Log("Matched")
        else:
            Log("Mismatched")
        return IfMatched

    def Test_ActivationTest(self,InputVector,ActualValue,CurrentID) -> bool :

        self._TestIV.Reset()
        self._TestOV.Reset()
        self._CreateConditionFactory.Reset()
        for N in self._MiddleLayerNodes:
            N.Reset()
        self._RootNode=RealNode("Root"+CurrentID)

        RootN=1000000
        RootActivation=self._RootNode.ActivateRoot(self._GlobalRootNode,1,RootN)
        self._TestIV.ActivateObservations(RootActivation,InputVector)

        self._TestIV.PrepareForTestPropagation()
        for N in self._MiddleLayerNodes:
            N.PrepareForTestPropagation()

        self._TestIV.PropagateValue()
        for N in self._MiddleLayerNodes:
            N.PropagateValue()
        ResultVector=self._TestOV.GetVectorResult()
        if LogLevel>=0:
            Log("InputVector ",InputVector," ResultVector ",ResultVector, " ReferenceVector ",ActualValue)
           
        Log("Predicted" , ResultVector)
        Log("Actual " ,ActualValue)
        IfMatched=True
        for j in range(len(ResultVector)):
            if (ResultVector[j]!=ActualValue[j]):
                IfMatched=False
        if (IfMatched):
            Log("Activation test all OK")
        else:
            Log("Activation test NG")
        return IfMatched

    def Test_AVecTest(self) -> bool:
        RootN=10000
        RootProbability=1
        RootS=RootN
        TestAVec0=AVec()
        ANode=RealNode("ANode")
        RootNode1=RealNode("TestRootNode1")
        RootNode2=RealNode("TestRootNode2")
        RootNode3=RealNode("TestRootNode3")
        RootNode4=RealNode("TestRootNode4")
        RootNode5=RealNode("TestRootNode5")
        RootActivation1=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode1,[],TestAVec0)
        RootActivation2=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode2,[],TestAVec0)
        RootActivation3=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode3,[],TestAVec0)
        RootActivation4=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode4,[],TestAVec0)
        RootActivation5=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode5,[],TestAVec0)

        TestAVecSet1=AVec()
        TestAVecSet1.AddSubstantialElement(RootActivation1)
        TestAVecSet1.AddSubstantialElement(RootActivation2)
 
        TestAVecSet2=AVec()
        TestAVecSet2.AddSubstantialElement(RootActivation3)
        TestAVecSet2.AddSubstantialElement(RootActivation1)

        TestAVecSet3=AVec()
        TestAVecSet3.AddSubstantialElement(RootActivation1)
        TestAVecSet3.AddSubstantialElement(RootActivation2)
        TestAVecSet3.AddSubstantialElement(RootActivation4)

        TestAVecSet4=AVec()
        TestAVecSet4.AddSubstantialElement(RootActivation1)
        TestAVecSet4.AddSubstantialElement(RootActivation5)
        TestAVecSet4.AddSubstantialElement(RootActivation4)
        
        IfMatched=True
        Log("Compare AVecs")
        Result0=TestAVecSet1.CompareSubstantialElements(TestAVecSet1)
        assert(Result0==SetCompareResult.Same)
        IfMatched =IfMatched and (Result0==SetCompareResult.Same)
        Result1=TestAVecSet3.CompareSubstantialElements(TestAVecSet1)
        assert(Result1==SetCompareResult.Contained)
        IfMatched =IfMatched and (Result1==SetCompareResult.Contained)
        Result2=TestAVecSet1.CompareSubstantialElements(TestAVecSet3)
        assert(Result2==SetCompareResult.Contain)
        IfMatched =IfMatched and (Result2==SetCompareResult.Contain)
        Result3=TestAVecSet1.CompareSubstantialElements(TestAVecSet2)
        assert(Result3==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (Result3==SetCompareResult.NoRelation)
        Result3=TestAVecSet2.CompareSubstantialElements(TestAVecSet3)
        assert(Result3==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (Result3==SetCompareResult.NoRelation)

        Log("Compare AVecs with series nodes")
        OriginNode1=RealNode("TimeNode1")
        SeriesNode1=SeriesNode(OriginNode1,0)
        SeriesNode2=SeriesNode(OriginNode1,4)
        SeriesNode3=SeriesNode(OriginNode1,35)
        RootActivation1S=Activation(RootProbability,RootProbability,RootN,None,ANode,SeriesNode1,[],TestAVec0)
        RootActivation2S=Activation(RootProbability,RootProbability,RootN,None,ANode,SeriesNode2,[],TestAVec0)
        RootActivation3S=Activation(RootProbability,RootProbability,RootN,None,ANode,SeriesNode3,[],TestAVec0)
        TestAVecSetS1=AVec()
        TestAVecSetS1.AddSubstantialElement(RootActivation1)
        TestAVecSetS1.AddSubstantialElement(RootActivation2)
        TestAVecSetS1.AddSubstantialElement(RootActivation1S)
        TestAVecSetS2=AVec()
        TestAVecSetS2.AddSubstantialElement(RootActivation1)
        TestAVecSetS2.AddSubstantialElement(RootActivation2)
        TestAVecSetS2.AddSubstantialElement(RootActivation2S)
        TestAVecSetS3=AVec()
        TestAVecSetS3.AddSubstantialElement(RootActivation1)
        TestAVecSetS3.AddSubstantialElement(RootActivation2)
        TestAVecSetS3.AddSubstantialElement(RootActivation3)
        TestAVecSetS3.AddSubstantialElement(RootActivation3S)
        TestAVecSetS4=AVec()
        TestAVecSetS4.AddSubstantialElement(RootActivation1)
        TestAVecSetS4.AddSubstantialElement(RootActivation2)
        TestAVecSetS4.AddSubstantialElement(RootActivation3)
        TestAVecSetS4.AddSubstantialElement(RootActivation1S)
        TestAVecSetS5=AVec()
        TestAVecSetS5.AddSubstantialElement(RootActivation1)
        TestAVecSetS5.AddSubstantialElement(RootActivation3)
        TestAVecSetS5.AddSubstantialElement(RootActivation2S)

        ResultS1=TestAVecSetS1.CompareSubstantialElements(TestAVecSetS2)
        assert(ResultS1==SetCompareResult.Contain)
        IfMatched =IfMatched and (ResultS1==SetCompareResult.Contain)
        ResultS2=TestAVecSetS1.CompareSubstantialElements(TestAVecSetS3)
        assert(ResultS2==SetCompareResult.Contain)
        IfMatched =IfMatched and (ResultS2==SetCompareResult.Contain)
        ResultS3=TestAVecSetS1.CompareSubstantialElements(TestAVecSetS4)
        assert(ResultS3==SetCompareResult.Contain)
        IfMatched =IfMatched and (ResultS3==SetCompareResult.Contain)
        ResultS4=TestAVecSetS1.CompareSubstantialElements(TestAVecSetS5)
        assert(ResultS4==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (ResultS4==SetCompareResult.NoRelation)
        ResultS5=TestAVecSetS2.CompareSubstantialElements(TestAVecSetS4)
        assert(ResultS5==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (ResultS5==SetCompareResult.NoRelation)

        ResultS1R=TestAVecSetS2.CompareSubstantialElements(TestAVecSetS1)
        assert(ResultS1R==SetCompareResult.Contained)
        IfMatched =IfMatched and (ResultS1R==SetCompareResult.Contained)
        ResultS2R=TestAVecSetS3.CompareSubstantialElements(TestAVecSetS1)
        assert(ResultS2R==SetCompareResult.Contained)
        IfMatched =IfMatched and (ResultS2R==SetCompareResult.Contained)
        ResultS3R=TestAVecSetS4.CompareSubstantialElements(TestAVecSetS1)
        assert(ResultS3R==SetCompareResult.Contained)
        IfMatched =IfMatched and (ResultS3R==SetCompareResult.Contained)
        ResultS4R=TestAVecSetS5.CompareSubstantialElements(TestAVecSetS1)
        assert(ResultS4R==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (ResultS4R==SetCompareResult.NoRelation)
        ResultS5R=TestAVecSetS4.CompareSubstantialElements(TestAVecSetS2)
        assert(ResultS5R==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (ResultS5R==SetCompareResult.NoRelation)

        Log("Compare complementary activations ")
        RootProbabilityBar=0
        RootActivation1R=Activation(RootProbability,RootProbabilityBar,RootN,None,ANode,RootNode1,[],TestAVec0)
        RootActivation2R=Activation(RootProbability,RootProbabilityBar,RootN,None,ANode,RootNode2,[],TestAVec0)
        RootActivation1RS=Activation(RootProbability,RootProbabilityBar,RootN,None,ANode,SeriesNode1,[],TestAVec0)
        RootActivation2RS=Activation(RootProbability,RootProbabilityBar,RootN,None,ANode,SeriesNode2,[],TestAVec0)
        RootActivation3RS=Activation(RootProbability,RootProbabilityBar,RootN,None,ANode,SeriesNode3,[],TestAVec0)
        TestAVecSetR1=AVec()
        TestAVecSetR1.AddSubstantialElement(RootActivation1)
        TestAVecSetR1.AddSubstantialElement(RootActivation2)
        TestAVecSetR1.AddSubstantialElement(RootActivation4)
        TestAVecSetR2=AVec()
        TestAVecSetR2.AddSubstantialElement(RootActivation1R)
        TestAVecSetR2.AddSubstantialElement(RootActivation2)
        TestAVecSetR2.AddSubstantialElement(RootActivation4)
        TestAVecSetR3=AVec()
        TestAVecSetR3.AddSubstantialElement(RootActivation1)
        TestAVecSetR3.AddSubstantialElement(RootActivation2R)
        TestAVecSetR3.AddSubstantialElement(RootActivation4)
        TestAVecSetR4=AVec()
        TestAVecSetR4.AddSubstantialElement(RootActivation1R)
        TestAVecSetR4.AddSubstantialElement(RootActivation2R)
        TestAVecSetR4.AddSubstantialElement(RootActivation4)
        TestAVecSetR5=AVec()
        TestAVecSetR5.AddSubstantialElement(RootActivation2)
        TestAVecSetR5.AddSubstantialElement(RootActivation4)
        TestAVecSetR5.AddSubstantialElement(RootActivation1S)
        TestAVecSetR6=AVec()
        TestAVecSetR6.AddSubstantialElement(RootActivation2)
        TestAVecSetR6.AddSubstantialElement(RootActivation4)
        TestAVecSetR6.AddSubstantialElement(RootActivation3S)
        TestAVecSetR7=AVec()
        TestAVecSetR7.AddSubstantialElement(RootActivation2)
        TestAVecSetR7.AddSubstantialElement(RootActivation4)
        TestAVecSetR7.AddSubstantialElement(RootActivation1RS)
        TestAVecSetR8=AVec()
        TestAVecSetR8.AddSubstantialElement(RootActivation2)
        TestAVecSetR8.AddSubstantialElement(RootActivation4)
        TestAVecSetR8.AddSubstantialElement(RootActivation3RS)

        ResultS1C=TestAVecSetR1.CompareSubstantialElements(TestAVecSetR2)
        assert(ResultS1C==SetCompareResult.Complementary)
        IfMatched =IfMatched and (ResultS1C==SetCompareResult.Complementary)
        ResultS2C=TestAVecSetR1.CompareSubstantialElements(TestAVecSetR3)
        assert(ResultS2C==SetCompareResult.Complementary)
        IfMatched =IfMatched and (ResultS2C==SetCompareResult.Complementary)
        ResultS3C=TestAVecSetR1.CompareSubstantialElements(TestAVecSetR4)
        assert(ResultS3C==SetCompareResult.Exclusive)
        IfMatched =IfMatched and (ResultS3C==SetCompareResult.Exclusive)
        ResultS4C=TestAVecSetR5.CompareSubstantialElements(TestAVecSetR7)
        assert(ResultS4C==SetCompareResult.Complementary)
        IfMatched =IfMatched and (ResultS4C==SetCompareResult.Complementary)
        ResultS5C=TestAVecSetR6.CompareSubstantialElements(TestAVecSetR7)
        assert(ResultS5C==SetCompareResult.Exclusive)
        IfMatched =IfMatched and (ResultS5C==SetCompareResult.Exclusive)
        ResultS6C=TestAVecSetR5.CompareSubstantialElements(TestAVecSetR8)
        assert(ResultS6C==SetCompareResult.NoRelation)
        IfMatched =IfMatched and (ResultS6C==SetCompareResult.NoRelation)

        if (IfMatched):
            Log("AVec test OK")
        else:
            Log("AVec test NG")
        return IfMatched
    
    def Test_ResetTestCounter(self):
        self.Test_IfMatched=True
        self.Test_SuccessCounter=0
        self.Test_TotalCounter=0

    def Test_Counter(self,Result:bool):
        self.Test_IfMatched&=Result
        self.Test_SuccessCounter+=Result
        self.Test_TotalCounter+=1
        if not Result:
           print("NG detected")
        #assert(Result)

    def Test_PWVPropagation(self) -> bool:
        # Create PWVs
        RootN=10000
        RootProbability=1
        RootS=RootN
        TestAVec0=AVec()
        ANode=RealNode("ANode")
        RootNode1=RealNode("TestRootNode1")
        RootNode2=RealNode("TestRootNode2")
        RootNode3=RealNode("TestRootNode3")
        RootNode4=RealNode("TestRootNode4")
        RootNode5=RealNode("TestRootNode5")
        RootNode6=RealNode("TestRootNode6")
        Equation=Polynomial([RootProbability])
        Activation1=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode1,[],TestAVec0)
        Activation2=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode2,[],TestAVec0)
        Activation3=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode3,[],TestAVec0)
        Activation4=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode4,[],TestAVec0)
        Activation5=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode5,[],TestAVec0)
        Activation6=Activation(RootProbability,RootProbability,RootN,None,ANode,RootNode6,[],TestAVec0)
        R1,E1,Weight1=Link.CalculateREWeight(0.6,1,True)  # random
        R1R,E1R,Weight1R=Link.CalculateREWeight(0.6,1,False)  # random
        R2,E2,Weight2=Link.CalculateREWeight(0.9,100,True)    #  <1 
        R2R,E2R,Weight2R=Link.CalculateREWeight(0.9,100,False)    # < 1 
        R3,E3,Weight3=Link.CalculateREWeight(1,100,True)     # ==1
        R4,E4,Weight4=Link.CalculateREWeight(0,1000,True)    # ==0
        R5,E5,Weight5=Link.CalculateREWeight(0.499,1000000,True)  # stable random
        R6,E6,Weight6=Link.CalculateREWeight(0.2,100,True)   # >0
        #R,E,Weight,P11,RootActivation
        PWV1=Propagation_weight_vector(1)    # P11=0.6  E=1
        PWV1.AddRootActivation(E1,Weight1,True,Activation1)
        PWV1.PropagateOnActivation(R1,E1*Weight1,1,0)
        PWV2=Propagation_weight_vector(1)    # P00=0.6  
        PWV2.AddRootActivation(E1R,Weight1R,False,Activation1)
        PWV2.PropagateOnActivation(0,0,R1R,E1R*Weight1R)
        PWV3=Propagation_weight_vector(1)    # P11=0.9 
        PWV3.AddRootActivation(E2,Weight2,True,Activation2)
        PWV3.PropagateOnActivation(R2,E2*Weight2,1,0)
        PWV4=Propagation_weight_vector(1)    # P00=0.9 
        PWV4.AddRootActivation(E2R,Weight2R,False,Activation2)
        PWV4.PropagateOnActivation(0,0,R2R,E2R*Weight2R)
        PWV5=Propagation_weight_vector(1)    # P11=1 
        PWV5.AddRootActivation(E3,Weight3,True,Activation3)
        PWV5.PropagateOnActivation(R3,E3*Weight3,1,0)
        PWV6=Propagation_weight_vector(1)    # P11=0 
        PWV6.AddRootActivation(E4,Weight4,True,Activation4)
        PWV6.PropagateOnActivation(R4,E4*Weight4,1,0)
        PWV7=Propagation_weight_vector(1)    # P11=0.5 
        PWV7.AddRootActivation(E5,Weight5,True,Activation5)
        PWV7.PropagateOnActivation(R5,E5*Weight5,1,0)
        PWV8=Propagation_weight_vector(1)    # P11=0.2
        PWV8.AddRootActivation(E6,Weight6,True,Activation6)
        PWV8.PropagateOnActivation(R6,E6*Weight6,1,0)
   
        self.Test_ResetTestCounter()

        print(" Propagate")
        CPWV1=PWV1.CreateClone()
        CPWV1.PropagateOnActivation(1,0,1,0) # P11=P00==1
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" Propagate NOT")
        CPWV1=PWV1.CreateClone()
        CPWV1.PropagateOnActivation(0,0,0,0) # P11=P00=0
        self.Test_Counter(CPWV1._Equation.coef[1]<0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E<0)
        # AND Operation
        print(" AND  1&1")
        CPWV1=PWV1.CreateClone()
        CPWV1.ANDOperation(PWV5)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" AND  1&0")
        CPWV1=PWV1.CreateClone()
        CPWV1.ANDOperation(PWV8)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._E>0)
        #print(" AND  1&0.5")
        CPWV1=PWV1.CreateClone()
        CPWV1.ANDOperation(PWV7)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._E>0)
        # OR Operation
        print(" OR  1|1")
        CPWV1=PWV1.CreateClone()
        CPWV1.OROperation(PWV5)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" OR  1|0")
        CPWV1=PWV1.CreateClone()
        CPWV1.OROperation(PWV8)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._E>0)
        print(" OR  0|0.5")
        CPWV1=PWV8.CreateClone()
        CPWV1.OROperation(PWV7)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation6)
        self.Test_Counter(PWO._E>0)
        print(" OR 0|0")
        CPWV1=PWV8.CreateClone()
        CPWV1.OROperation(PWV6)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation6)
        self.Test_Counter(PWO._E>0)
        print(" OR 0|1")
        CPWV1=PWV8.CreateClone()
        CPWV1.OROperation(PWV1)
        self.Test_Counter(CPWV1._Equation.coef[2]<0)
        PWO=CPWV1._GetPWOFromActivation(Activation6)
        self.Test_Counter(PWO._E>0)

        # XOR Operation
        print(" XOR  1^1")
        CPWV1=PWV1.CreateClone()
        CPWV1.XOROperation(PWV3)
        self.Test_Counter(CPWV1._Equation.coef[1]<0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E<0)
        print(" XOR  1^0")
        CPWV1=PWV1.CreateClone()
        CPWV1.XOROperation(PWV8)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" XOR  0^0.5")
        CPWV1=PWV8.CreateClone()
        CPWV1.XOROperation(PWV7)
        self.Test_Counter(CPWV1._Equation.coef[1]>0) #?
        PWO=CPWV1._GetPWOFromActivation(Activation6)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" XOR  1^0.5")
        CPWV1=PWV1.CreateClone()
        CPWV1.XOROperation(PWV7)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation1)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)
        print(" XOR 0^0")
        CPWV1=PWV7.CreateClone()
        CPWV1.XOROperation(PWV8)
        self.Test_Counter(CPWV1._Equation.coef[1]>0)
        PWO=CPWV1._GetPWOFromActivation(Activation5)
        self.Test_Counter(PWO._LinkWeight>0)
        self.Test_Counter(PWO._E>0)


        if (self.Test_IfMatched):
            Log("PWV test OK")
        else:
            Log("PWV test NG")
        Log(" Test success ratio ",self.Test_SuccessCounter,"/",self.Test_TotalCounter)
        return self.Test_IfMatched
        
    def Test_Feedback(self):
        # PWV Propagation + Create Link
        pass

    def Test_Reset(self):
        self._TestIV=None
        self._TestOV=None
        self._MiddleLayerNodes=[]

    def Test(self):
        #
        #  Propagaation test
        g_SOLNetwork.Test_CreateTestLogic()
        TestResult=True
        if True: # Temporary comment out
            Log("##############################################")
            Log("  Value propagation test")
            Log("##############################################")
            LocalTestResult=True
            for num in range(C_LearningIterations):
                i=g_random.randint(0,len(self._TestX)-1)
                LocalTestResult&=self.Test_ValuePropagationTest(self._TestX[i], self._TestY[i],"_"+str(i))
            TestResult&=LocalTestResult
            Log("Value propagation test all result :",LocalTestResult)
            Log("##############################################")
            Log("  Activation test")
            Log("##############################################")
            LocalTestResult=True
            for num in range(C_LearningIterations):
                i=g_random.randint(0,len(self._TestX)-1)
                TestResult&=self.Test_ActivationTest(self._TestX[i], self._TestY[i],"_"+str(i))
            TestResult&=LocalTestResult
            Log("Activation test all result :",LocalTestResult)

            Log("##############################################")
            Log("  AVec test")
            Log("##############################################")
            TestResult&=g_SOLNetwork.Test_AVecTest()

            Log("##############################################")
            Log("  PWV test")
            Log("##############################################")
            TestResult&=g_SOLNetwork.Test_PWVPropagation()

        Log("##############################################")
        Log("  Feedback test")
        Log("##############################################")
        g_SOLNetwork.Test_Feedback()

        g_SOLNetwork.Test_Reset()
        #
        #
        #
        if TestResult:
            Log("All test OK")
        else:
            Log("Test NG")



##########################################################################
# Data generator
##########################################################################

# Generate all combinations of data

if g_IfHugeInputMatrix:
    XSize=32
    X=SOLNetwork.GenerateMatrix(XSize,C_LearningIterations)
    X = torch.tensor(X, dtype=torch.float32)
    X_int = X.int()
else:
    XSize=4
    X = list(itertools.product([0, 1], repeat=XSize))

    X = torch.tensor(X, dtype=torch.float32)
    X_int = X.int()


Log('Inputs: ')
Log(X_int)


# Logic operations
YEquation=[]
YEquation.append(X_int[:, 0] & X_int[:, 1])
YEquation.append(X_int[:, 2] | X_int[:, 3])
YEquation.append(X_int[:, 1] ^ X_int[:, 3])
YEquation.append((~X_int[:, 0]&1) & X_int[:, 1])
YEquation.append((~(X_int[:, 0]& X_int[:, 1])&1)) 
YEquation.append((~X_int[:, 2]&1) | X_int[:, 3])
YEquation.append((~X_int[:, 0]&1) ^ X_int[:, 1])
YEquation.append((~(X_int[:, 2]& X_int[:, 3])&1)) 
YEquation.append((X_int[:, 0] & X_int[:, 1] & X_int[:, 2] & X_int[:, 3]))
YEquation.append((X_int[:, 0] | X_int[:, 1] | X_int[:, 2] | X_int[:, 3]))
YEquation.append((X_int[:, 0] ^ X_int[:, 1] ^ (~X_int[:, 2]&1) ^ X_int[:, 3]))
YEquation.append((X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3]))
YEquation.append((X_int[:, 0] | (~X_int[:, 1])&1) | (X_int[:, 2] | X_int[:, 3]))
YEquation.append((X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3]))
YEquation.append((X_int[:, 0] ^ (~X_int[:, 1]&1)) & (X_int[:, 2] ^ (~X_int[:, 3]&1)))
YEquation.append((X_int[:, 0] | X_int[:, 1]) ^ ((~X_int[:, 2]&1) & X_int[:, 3]))
ReferenceEq=[
"X_int[:, 0] & X_int[:, 1]",
"X_int[:, 2] | X_int[:, 3]",
"X_int[:, 1] ^ X_int[:, 3]",
"(~X_int[:, 0]&1) & X_int[:, 1]",
"(~(X_int[:, 0]& X_int[:, 1])&1)",
"(~X_int[:, 2]&1) | X_int[:, 3]",
"(~X_int[:, 0]&1) ^ X_int[:, 1]",
"~(X_int[:, 2]& X_int[:, 3])&1) ",
"X_int[:, 0] & X_int[:, 1] & X_int[:, 2] & X_int[:, 3]",
"X_int[:, 0] | X_int[:, 1] | X_int[:, 2] | X_int[:, 3]",
"X_int[:, 0] ^ X_int[:, 1] ^ (~X_int[:, 2]&1) ^ X_int[:, 3]",
"(X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3])",
"X_int[:, 0] | (~X_int[:, 1])&1) | (X_int[:, 2] | X_int[:, 3]",
"(X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3])",
"(X_int[:, 0] ^ (~X_int[:, 1]&1)) & (X_int[:, 2] ^ (~X_int[:, 3]&1))",
"(X_int[:, 0] | X_int[:, 1]) ^ ((~X_int[:, 2]&1) & X_int[:, 3])",
]
YVector=[]
for eachY in YEquation:
    YVector.append(eachY.float())

Y = torch.stack(YVector, dim=1)
YSize=Y.size(1)
#YSize=12 # Temporary

##########################################################################
# Start learning
##########################################################################
g_SOLNetwork=SOLNetwork(XSize,YSize)

if g_IfUnitTestAll:
    g_SOLNetwork.Test()
    exit()

start_time = time.time()

### Learning 
#import random
for num in range(C_LearningIterations):
    if g_IfHugeInputMatrix:
        i=num
    else:
        i=g_random.randint(0,len(X)-1)

    g_step=num+1
    if LogLevel==0:
        if g_step%100==0:
            Log("--- Learning step",g_step)
    else:
        Log("--- Learning data combination [",i,"] : ",g_step)

    if (g_step==5):
        Log("DEBUG: probe position step5")
        #LogLevel=2

    g_SOLNetwork.Learning(X[i], Y[i],g_step)
    if LogLevel>0:
        if g_step%10==0:
            g_diagnostics.DumpCurrentStatus()
    g_diagnostics.StepReset()

Log("--------------------------------------------")
Log("Learning ended")

end_time = time.time()

Log("------------------------------------------------")
Log(" Dump final equation (definitive value)")
Log("   N: Number of confirmations")

g_SOLNetwork.DumpEquation()
g_Dumpequationall=False
if g_Dumpequationall:
    Log("--------------------------------------------")
    Log(" Dump all random links ")
    g_SOLNetwork.DumpEquation(True)
    Log("--------------------------------------------")

##########################################################################
# Confirm learned results
##########################################################################
Matched=0
LineLength=YSize
MatchVector=np.zeros(LineLength,dtype=int)
TestCombinationSize=len(X)
for i in range(TestCombinationSize):
    g_SOLNetwork.Reset()
    Log(i,"Input" , X[i])
    ResultVector=g_SOLNetwork.GetResult(X[i])
    ActualValue=Y[i]
    Log(i,"Predicted" , ResultVector)
    Log(i,"Reference " ,ActualValue)
    MatchResult=[]
    for j in range(LineLength):
        if (abs(ResultVector[j]-ActualValue[j])<0.1):
            MatchVector[j]+=1
            MatchResult.insert(j,str(j)+":True")
        else:
            MatchResult.insert(j,str(j)+":False")

    Log("Compare result",MatchResult)

Log("--------------------------------------------")
MatchedLogics=0
for j in range(LineLength):
    if MatchVector[j]==TestCombinationSize:
        Log("Matched : Output",j," = ",ReferenceEq[j], sep='')
        MatchedLogics+=1
    else:
        Log("Mismatched : Output",j," = ",ReferenceEq[j], sep='')
        Equation=g_SOLNetwork._OutputVector.DumpEquationTree(j,0,False,True)
        #Equation=g_SOLNetwork._OutputVector.DumpEquationTree(j,0,True,True) # all dump
        Log(" Learned result ",Equation) 
Log("Learning result : matched logics " ,MatchedLogics, " / " ,len(ResultVector) )

elapsed_time = end_time - start_time
Log("Random seed" ,C_RandomSeed)
Log("Learning steps ",C_LearningIterations)
Log(f"Elapsed time of learning: {elapsed_time}sec")
Log("Current time",datetime.now()," Version",C_Version)
Log("Log file created to",log_file_path)

