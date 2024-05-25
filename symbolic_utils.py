from sklearn.metrics import accuracy_score
import numpy as np
import ltn
import torch

# we define the constants for classes
l_Hispanic    = ltn.Constant(torch.tensor([1, 0, 0, 0]))
l_Gothic      = ltn.Constant(torch.tensor([0, 1, 0, 0]))
l_Renaissance = ltn.Constant(torch.tensor([0, 0, 1, 0]))
l_Baroque     = ltn.Constant(torch.tensor([0, 0, 0, 1]))
# define constants for attributes  
l_arco_apuntado      = ltn.Constant(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_arco_conopial      = ltn.Constant(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_arco_herradura     = ltn.Constant(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_arco_lobulado      = ltn.Constant(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_arco_medio_punto   = ltn.Constant(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_arco_trilobulado   = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
l_columna_salomonica = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
l_dintel_adovelado   = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
l_fronton            = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
l_fronton_curvo      = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
l_fronton_partido    = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
l_ojo_de_buey        = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
l_pinaculo_gotico    = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
l_serliana           = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
l_vano_adintelado    = ltn.Constant(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

i_arco_apuntado      = 0
i_arco_conopial      = 1
i_arco_herradura     = 2
i_arco_lobulado      = 3
i_arco_medio_punto   = 4
i_arco_trilobulado   = 5
i_columna_salomonica = 6
i_dintel_adovelado   = 7
i_fronton            = 8
i_fronton_curvo      = 9
i_fronton_partido    = 10
i_ojo_de_buey        = 11
i_pinaculo_gotico    = 12
i_serliana           = 13
i_vano_adintelado    = 14


# we define the connectives, quantifiers, and the SatAgg
SatAgg = ltn.fuzzy_ops.SatAgg()
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
ForallP4 = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
ForallP6 = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6), quantifier="f")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
ForallP1 = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=1), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")




class LogitsToPredicateMultilabel(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """
    def __init__(self, logits_model):
        super(LogitsToPredicateMultilabel, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.sigmoid(logits)
        out = torch.sum(probs * l, dim=1)
        return out
    
class LogitsToPredicateMulticlass(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """
    def __init__(self, logits_model):
        super(LogitsToPredicateMulticlass, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()
        self.penalty = None
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x, l_class, tree_penalty=False):
        input = torch.nn.Sigmoid()(x)       
        
        if tree_penalty:
            final_logits_outputs, penalty = self.logits_model.forward(input,is_training_data=True)
            if self.penalty == None:
                self.penalty = penalty
            else:
                self.penalty += penalty            
        else:
            final_logits_outputs = self.logits_model.forward(input,is_training_data=False)

        probs = self.softmax(final_logits_outputs)

        if l_class is not None: # if is not None, we output only the prob for the choosen class, this is needed for the predicate
            return torch.sum(probs * l_class, dim=1)           
        else:
            raise Exception("This model is thought to be use in a predicate for an especific class")
        
    def get_penalty(self):
        return self.penalty
    
    def reset_penalty(self):
        self.penalty = None

class LogitsToPredicateResnetTree(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """
       
    def __init__(self, model1, model2):
        super(LogitsToPredicateResnetTree, self).__init__()
        self.first_model = model1
        self.second_model = model2
        self.softmax = torch.nn.Softmax(dim=1)
        self.penalty = None

    def forward_stage2(self, stage1_out, l_class = None, tree_penalty = False):
        # if  self.use_sigmoid:
        attr_outputs = torch.nn.Sigmoid()(stage1_out)
        # else:
        #     attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        
        if tree_penalty:
            final_logits_outputs, penalty = self.second_model.forward(stage2_inputs,is_training_data=tree_penalty)
            if self.penalty == None:
                self.penalty = penalty
            else:
                self.penalty += penalty            
        else:
            final_logits_outputs = self.second_model.forward(stage2_inputs,is_training_data=tree_penalty)

        probs = self.softmax(final_logits_outputs)

        if l_class is not None: # if is not None, we output only the prob for the choosen class, this is needed for the predicate
            return torch.sum(probs * l_class, dim=1)           
        
        return attr_outputs, probs

    def forward(self, x, l_class = None, tree_penalty=False):
        if self.first_model.training:
            outputs= self.first_model(x)
            return self.forward_stage2(outputs,l_class, tree_penalty)
        else:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs,l_class, tree_penalty=tree_penalty)
        
    def get_penalty(self):
        return self.penalty
    
    def reset_penalty(self):
        self.penalty = None

def get_axioms_classes(data,labels,tree_penalty,P):
    axioms = []
    ######################## add labels knowledge
    # axioms for hispanic if there are samples in the batch
    index_x_hispanic = data[labels == 0]        
    if len(index_x_hispanic):            
        x_Hispanic = ltn.Variable("x_Hispanic", index_x_hispanic) # class hispanic examples
        axioms.append(Forall(x_Hispanic, P(x_Hispanic, l_Hispanic, tree_penalty=tree_penalty)))
    
    # axioms for gothic if there are samples in the batch
    index_x_gothic = data[labels == 1]            
    if len(index_x_gothic):            
        x_Gothic = ltn.Variable("x_Gothic", index_x_gothic) # class gothic examples
        axioms.append(Forall(x_Gothic,P(x_Gothic,l_Gothic,tree_penalty = tree_penalty)))
    
    # axioms for renaissance if there are samples in the batch
    index_x_renaissance = data[labels == 2]        
    if len(index_x_renaissance):           
        x_Renaissance = ltn.Variable("x_Renaissance", index_x_renaissance) # class Renaissance examples
        axioms.append(Forall(x_Renaissance, P(x_Renaissance, l_Renaissance, tree_penalty=tree_penalty)))
    
    # axioms for baroque if there are samples in the batch
    index_x_baroque = data[labels == 3]   
    if len(index_x_baroque):            
        x_Baroque = ltn.Variable("x_Baroque", index_x_baroque) # class Barque examples
        axioms.append( Forall(x_Baroque, P(x_Baroque, l_Baroque, tree_penalty=tree_penalty)))
    return axioms

def get_axioms_attr_joint(data,tree_penalty,P_attr,P_full):
    axioms = []
    # add attributes knowledge  ESTO HAY QUE DIVIDIRLO EN 4 y computar 4 sat separados con 4 axioms_list distintos!             
    x = ltn.Variable("x", data)
    # FALTA garantizar que los atricutos sean correctamente detectados!
    P_full_gothic = P_full(x, l_Gothic,tree_penalty=tree_penalty)
    P_full_hispanic = P_full(x, l_Hispanic,tree_penalty=tree_penalty)
    P_full_baroque = P_full(x, l_Baroque,tree_penalty=tree_penalty)
    P_full_renaissance = P_full(x, l_Renaissance, tree_penalty=tree_penalty)
    #x_attr = ltn.Variable ("x_attr",y) hay que encontrar como pasar de x a y 
    axioms.append(Forall(x, Implies(P_attr(x, l_arco_apuntado), P_full_gothic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_arco_conopial),P_full_gothic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_arco_herradura), P_full_hispanic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_arco_lobulado), P_full_hispanic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_arco_trilobulado), P_full_gothic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_columna_salomonica),  P_full_baroque), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_dintel_adovelado), P_full_hispanic), p=5))
    axioms.append(Forall(x, Implies(P_attr(x, l_fronton_curvo),Not(P_full_hispanic)), p=5)) 
    axioms.append(Forall(x, Implies(P_attr(x, l_ojo_de_buey),And(Not(P_full_hispanic),Not(P_full_gothic)), p=5)))
    axioms.append(Forall(x, Implies(P_attr(x, l_pinaculo_gotico),P_full_gothic), p=5)) # aqui puede ser renaissance en test
    axioms.append(Forall(x, Implies(P_attr(x, l_serliana), Or(P_full_renaissance,P_full_baroque), p=5)))
    axioms.append(Forall(x, Implies(P_attr(x, l_vano_adintelado),Or(P_full_baroque,P_full_renaissance)), p=5)) # este en test si es asi, aunque en training y val hay algunos 
    
    return axioms 

def get_axioms_attr_guarded(attr_outputs,tree_penalty,P_class):
    axioms = []
    # add attributes knowledge  ESTO HAY QUE DIVIDIRLO EN 4 y computar 4 sat separados con 4 axioms_list distintos!             
    x = ltn.Variable("attr_outputs", attr_outputs)
    #x_attr = ltn.Variable ("x_attr",y) hay que encontrar como pasar de x a y 
    axioms.append(Forall([x], P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_apuntado.value, dim=1),0.5))) # aqui hacer un  guarded quantification  Forall_arco_apuntado https://github.com/logictensornetworks/LTNtorch/blob/db64c39a5d75b084ddff2524b26368a37c6584f3/tutorials/2-grounding_connectives.ipynb
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_conopial.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_herradura.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_lobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_trilobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Baroque,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_columna_salomonica.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_dintel_adovelado.value, dim=1),0.5)))
    axioms.append(Forall(x, Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_fronton_curvo.value, dim=1),0.5)))
    axioms.append(Forall(x, And(Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)),Not(P_class(x,l_Gothic,tree_penalty=tree_penalty))), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_ojo_de_buey.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_pinaculo_gotico.value, dim=1),0.5)))  # aqui puede ser renaissance en test
    axioms.append(Forall(x, Or(P_class(x,l_Renaissance,tree_penalty=tree_penalty),P_class(x,l_Baroque,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_serliana.value, dim=1),0.5)))
    axioms.append(Forall(x,Or(P_class(x,l_Baroque,tree_penalty=tree_penalty),P_class(x,l_Renaissance,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_vano_adintelado.value, dim=1),0.5))) # este en test si es asi, aunque en training y val hay algunos 
    
    return axioms 

def get_axioms_attr_guarded_sigmoid(attr_outputs,tree_penalty,P_class):
    axioms = []
    # add attributes knowledge  ESTO HAY QUE DIVIDIRLO EN 4 y computar 4 sat separados con 4 axioms_list distintos!             
    x = ltn.Variable("attr_outputs", attr_outputs)
    #x_attr = ltn.Variable ("x_attr",y) hay que encontrar como pasar de x a y 
    axioms.append(Forall([x], P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_arco_apuntado.value, dim=1),0.5))) # aqui hacer un  guarded quantification  Forall_arco_apuntado https://github.com/logictensornetworks/LTNtorch/blob/db64c39a5d75b084ddff2524b26368a37c6584f3/tutorials/2-grounding_connectives.ipynb
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( (x.value) * l_arco_conopial.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( (x.value) * l_arco_herradura.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( (x.value) * l_arco_lobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_arco_trilobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Baroque,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_columna_salomonica.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_dintel_adovelado.value, dim=1),0.5)))
    axioms.append(Forall(x, Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_fronton_curvo.value, dim=1),0.5)))
    axioms.append(Forall(x, And(Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)),Not(P_class(x,l_Gothic,tree_penalty=tree_penalty))), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_ojo_de_buey.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_pinaculo_gotico.value, dim=1),0.5)))  # aqui puede ser renaissance en test
    axioms.append(Forall(x, Or(P_class(x,l_Renaissance,tree_penalty=tree_penalty),P_class(x,l_Baroque,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_serliana.value, dim=1),0.5)))
    axioms.append(Forall(x,Or(P_class(x,l_Baroque,tree_penalty=tree_penalty),P_class(x,l_Renaissance,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( (x.value) * l_vano_adintelado.value, dim=1),0.5))) # este en test si es asi, aunque en training y val hay algunos 
    
    return axioms 
def get_axioms_attr_guarded_and_class(attr_outputs,labels,tree_penalty,P_class):
    axioms = []
    # add attributes knowledge  ESTO HAY QUE DIVIDIRLO EN 4 y computar 4 sat separados con 4 axioms_list distintos!             
    x = ltn.Variable("attr_outputs", attr_outputs)
    
    axioms.append(Forall([x], P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_apuntado.value, dim=1),0.5))) # aqui hacer un  guarded quantification  Forall_arco_apuntado https://github.com/logictensornetworks/LTNtorch/blob/db64c39a5d75b084ddff2524b26368a37c6584f3/tutorials/2-grounding_connectives.ipynb
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_conopial.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_herradura.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x:  torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_lobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_arco_trilobulado.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Baroque,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_columna_salomonica.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_dintel_adovelado.value, dim=1),0.5)))
    axioms.append(Forall(x, Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_fronton_curvo.value, dim=1),0.5)))
    axioms.append(Forall(x, And(Not(P_class(x,l_Hispanic,tree_penalty=tree_penalty)),Not(P_class(x,l_Gothic,tree_penalty=tree_penalty))), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_ojo_de_buey.value, dim=1),0.5)))
    axioms.append(Forall(x, P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_pinaculo_gotico.value, dim=1),0.5)))  # aqui puede ser renaissance en test
    axioms.append(Forall(x, Or(P_class(x,l_Renaissance,tree_penalty=tree_penalty),P_class(x,l_Baroque,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_serliana.value, dim=1),0.5)))
    axioms.append(Forall(x,Or(P_class(x,l_Baroque,tree_penalty=tree_penalty),P_class(x,l_Renaissance,tree_penalty=tree_penalty)), cond_vars=[x], cond_fn = lambda x: torch.greater_equal(torch.sum( torch.nn.Sigmoid()(x.value) * l_vano_adintelado.value, dim=1),0.5))) # este en test si es asi, aunque en training y val hay algunos 
    labels = torch.nn.functional.one_hot(labels,num_classes= 4)#.to(torch.float)
    y = ltn.Variable ("labels",labels) 
    axioms.append(ForallP6(ltn.diag(x,y), P_class(x,l_Gothic,tree_penalty=tree_penalty), cond_vars=[y], cond_fn = lambda y: torch.greater_equal(torch.sum( (y.value) * l_Gothic.value, dim=1),0.5)))
    axioms.append(ForallP6(ltn.diag(x,y), P_class(x,l_Hispanic,tree_penalty=tree_penalty), cond_vars=[y], cond_fn = lambda y: torch.greater_equal(torch.sum( (y.value) * l_Hispanic.value, dim=1),0.5)))
    axioms.append(ForallP6(ltn.diag(x,y), P_class(x,l_Renaissance,tree_penalty=tree_penalty), cond_vars=[y], cond_fn = lambda y: torch.greater_equal(torch.sum( (y.value) * l_Renaissance.value, dim=1),0.5)))
    axioms.append(ForallP6(ltn.diag(x,y), P_class(x,l_Baroque,tree_penalty=tree_penalty), cond_vars=[y], cond_fn = lambda y: torch.greater_equal(torch.sum( (y.value) * l_Baroque.value, dim=1),0.5)))
    return axioms 
def get_axioms_attr_multilabel(data,labels,is_training,P):
    axioms = []
    # add attributes knowledge            
    index_x_arco_apuntado = data[labels[:,0]==1]            
    if len(index_x_arco_apuntado):                      
        x_arco_apuntado = ltn.Variable("x_arco_apuntado", index_x_arco_apuntado) 
        axioms.append(Forall(x_arco_apuntado, P(x_arco_apuntado, l_arco_apuntado, is_training=is_training))) 

    index_x_arco_conopial = data[labels[:,1]==1]        
    if len(index_x_arco_conopial):            
        x_arco_conopial = ltn.Variable("x_arco_conopial", index_x_arco_conopial) 
        axioms.append(Forall(x_arco_conopial, P(x_arco_conopial, l_arco_conopial, is_training=is_training))) 
        
    index_x_arco_herradura = data[labels[:,2]==1]        
    if len(index_x_arco_herradura):            
        x_arco_herradura = ltn.Variable("x_arco_herradura", index_x_arco_herradura) 
        axioms.append(Forall(x_arco_herradura, P(x_arco_herradura, l_arco_herradura, is_training=is_training))) 
    
    index_x_arco_lobulado = data[labels[:,3]==1]        
    if len(index_x_arco_lobulado):            
        x_arco_lobulado = ltn.Variable("x_arco_lobulado", index_x_arco_lobulado) 
        axioms.append(Forall(x_arco_lobulado, P(x_arco_lobulado, l_arco_lobulado, is_training=is_training))) 

    index_x_arco_medio_punto = data[labels[:,4]==1]        # este tb puede ser de todo
    if len(index_x_arco_medio_punto):            
        x_arco_medio_punto = ltn.Variable("x_arco_medio_punto", index_x_arco_medio_punto) # class hispanic examples
        axioms.append(Forall(x_arco_medio_punto, P(x_arco_medio_punto, l_arco_medio_punto, is_training=is_training)))     
    
    index_x_arco_trilobulado = data[labels[:,5]==1]        
    if len(index_x_arco_trilobulado):            
        x_arco_trilobulado = ltn.Variable("x_arco_trilobulado", index_x_arco_trilobulado) 
        axioms.append(Forall(x_arco_trilobulado, P(x_arco_trilobulado, l_arco_trilobulado, is_training=is_training))) 
    
    index_x_columna_salomonica = data[labels[:,6]==1]        
    if len(index_x_columna_salomonica):            
        x_columna_salomonica = ltn.Variable("x_columna_salomonica", index_x_arco_trilobulado) 
        axioms.append(Forall(x_columna_salomonica, P(x_columna_salomonica, l_arco_trilobulado, is_training=is_training)))

    index_x_dintel_adovelado = data[labels[:,7]==1]        
    if len(index_x_dintel_adovelado):            
        x_dintel_adovelado = ltn.Variable("x_dintel_adovelado", index_x_arco_trilobulado) 
        axioms.append(Forall(x_dintel_adovelado, P(x_dintel_adovelado, l_arco_trilobulado, is_training=is_training))) 

    index_x_fronton = data[labels[:,8]==1]        
    if len(index_x_fronton):            
        x_fronton = ltn.Variable("x_fronton", index_x_fronton) 
        axioms.append(Forall(x_fronton, P(x_fronton, l_fronton, is_training=is_training)))

    index_x_fronton_curvo = data[labels[:,9]==1]        # Aqui hay que tener cuidado porque ademas puede ser gotico
    if len(index_x_fronton_curvo):            
        x_fronton_curvo = ltn.Variable("x_fronton_curvo", index_x_fronton_curvo) 
        axioms.append(Forall(x_fronton_curvo, Not(P(x_fronton_curvo, l_fronton_curvo, is_training=is_training))))

    index_x_fronton_partido = data[labels[:,10]==1]        
    if len(index_x_fronton_partido):            
        x_fronton_partido = ltn.Variable("x_fronton_partido", index_x_fronton_partido) 
        axioms.append(Forall(x_fronton_partido, P(x_fronton_partido, l_fronton_partido, is_training=is_training)))

    index_x_ojo_de_buey = data[labels[:,11]==1]        # El ojo de buey tb puede ser gotico
    if len(index_x_ojo_de_buey):            
        x_ojo_de_buey = ltn.Variable("x_ojo_de_buey", index_x_ojo_de_buey) 
        axioms.append(Forall(x_ojo_de_buey, Not(P(x_ojo_de_buey, l_ojo_de_buey, is_training=is_training))))
    
    index_x_pinaculo_gotico = data[labels[:,12]==1]        
    if len(index_x_pinaculo_gotico):            
        x_pinaculo_gotico = ltn.Variable("x_pinaculo_gotico", index_x_pinaculo_gotico) 
        axioms.append(Forall(x_pinaculo_gotico, P(x_pinaculo_gotico, l_pinaculo_gotico, is_training=is_training)))
    
    index_x_serliana = data[labels[:,13]==1]        
    if len(index_x_serliana):            
        x_serliana = ltn.Variable("x_serliana", index_x_serliana) 
        axioms.append(Forall(x_serliana, P(x_serliana, l_serliana, is_training=is_training)))

    index_x_vano_adintelado = data[labels[:,14]==1]        
    if len(index_x_vano_adintelado):            
        x_vano_adintelado = ltn.Variable("x_vano_adintelado", index_x_vano_adintelado) 
        axioms.append(ForallP1(x_vano_adintelado, P(x_vano_adintelado, l_vano_adintelado, is_training=is_training)))
    ############################################
    return axioms 

def get_axioms_attr(data,labels,tree_penalty,P):
    axioms = []
    # add attributes knowledge              HERE WE DO NOT COMPUTE TREE PENALTY AS IT WAS ALREADY COMPUTE FOR ALL THE SAMPLES BEFORE
    # axioms for arco_apuntado if there are samples in the batch
    index_x_arco_apuntado = data[labels[:,0]==1]            
    if len(index_x_arco_apuntado):                      
        x_arco_apuntado = ltn.Variable("x_arco_apuntado", index_x_arco_apuntado) 
        axioms.append(Forall(x_arco_apuntado, P(x_arco_apuntado, l_Gothic, tree_penalty=tree_penalty))) 

    index_x_arco_conopial = data[labels[:,1]==1]        
    if len(index_x_arco_conopial):            
        x_arco_conopial = ltn.Variable("x_arco_conopial", index_x_arco_conopial) 
        axioms.append(Forall(x_arco_conopial, P(x_arco_conopial, l_Gothic, tree_penalty=tree_penalty))) 
        
    index_x_arco_herradura = data[labels[:,2]==1]        
    if len(index_x_arco_herradura):            
        x_arco_herradura = ltn.Variable("x_arco_herradura", index_x_arco_herradura) 
        axioms.append(Forall(x_arco_herradura, P(x_arco_herradura, l_Hispanic, tree_penalty=tree_penalty))) 
    
    index_x_arco_lobulado = data[labels[:,3]==1]        
    if len(index_x_arco_lobulado):            
        x_arco_lobulado = ltn.Variable("x_arco_lobulado", index_x_arco_lobulado) 
        axioms.append(Forall(x_arco_lobulado, P(x_arco_lobulado, l_Hispanic, tree_penalty=tree_penalty))) 

    # index_x_arco_medio_punto = data[labels[:,4]==1]        # este tb puede ser de todo
    # if len(index_x_arco_medio_punto):            
    #     x_arco_medio_punto = ltn.Variable("x_arco_medio_punto", index_x_arco_medio_punto) # class hispanic examples
    #     axioms.append(Forall(x_arco_medio_punto, Or(P(x_arco_medio_punto, l_Baroque, tree_penalty=tree_penalty), P(x_arco_medio_punto, l_Renaissance, tree_penalty=tree_penalty))))     
    
    index_x_arco_trilobulado = data[labels[:,5]==1]        
    if len(index_x_arco_trilobulado):            
        x_arco_trilobulado = ltn.Variable("x_arco_trilobulado", index_x_arco_trilobulado) 
        axioms.append(Forall(x_arco_trilobulado, P(x_arco_trilobulado, l_Gothic, tree_penalty=tree_penalty))) 
    
    index_x_columna_salomonica = data[labels[:,6]==1]        
    if len(index_x_columna_salomonica):            
        x_columna_salomonica = ltn.Variable("x_columna_salomonica", index_x_columna_salomonica) 
        axioms.append(Forall(x_columna_salomonica, P(x_columna_salomonica, l_Baroque, tree_penalty=tree_penalty)))

    index_x_dintel_adovelado = data[labels[:,7]==1]        
    if len(index_x_dintel_adovelado):            
        x_dintel_adovelado = ltn.Variable("x_dintel_adovelado", index_x_dintel_adovelado) 
        axioms.append(Forall(x_dintel_adovelado, P(x_dintel_adovelado, l_Hispanic, tree_penalty=tree_penalty))) 

    # index_x_fronton = data[labels[:,8]==1]        
    # if len(index_x_fronton):            
    #     x_fronton = ltn.Variable("x_fronton", index_x_fronton) 
    #     axioms.append(Forall(x_fronton, P(x_fronton, l_Renaissance, tree_penalty=tree_penalty)))

    index_x_fronton_curvo = data[labels[:,9]==1]        # Aqui hay que tener cuidado porque ademas puede ser gotico
    if len(index_x_fronton_curvo):            
        x_fronton_curvo = ltn.Variable("x_fronton_curvo", index_x_fronton_curvo) 
        axioms.append(Forall(x_fronton_curvo, Not(P(x_fronton_curvo, l_Hispanic, tree_penalty=tree_penalty))))

    # index_x_fronton_partido = data[labels[:,10]==1]        
    # if len(index_x_fronton_partido):            
    #     x_fronton_partido = ltn.Variable("x_fronton_partido", index_x_fronton_partido) 
    #     axioms.append(Forall(x_fronton_partido, P(x_fronton_partido, l_Baroque, tree_penalty=tree_penalty)))

    index_x_ojo_de_buey = data[labels[:,11]==1]        # El ojo de buey tb puede ser gotico
    if len(index_x_ojo_de_buey):            
        x_ojo_de_buey = ltn.Variable("x_ojo_de_buey", index_x_ojo_de_buey) 
        axioms.append(Forall(x_ojo_de_buey, Not(P(x_ojo_de_buey, l_Hispanic, tree_penalty=tree_penalty))))
    
    index_x_pinaculo_gotico = data[labels[:,12]==1]        
    if len(index_x_pinaculo_gotico):            
        x_pinaculo_gotico = ltn.Variable("x_pinaculo_gotico", index_x_pinaculo_gotico) 
        axioms.append(Forall(x_pinaculo_gotico, P(x_pinaculo_gotico, l_Gothic, tree_penalty=tree_penalty)))
    
    index_x_serliana = data[labels[:,13]==1]        
    if len(index_x_serliana):            
        x_serliana = ltn.Variable("x_serliana", index_x_serliana) 
        axioms.append(Forall(x_serliana, Or(P(x_serliana, l_Renaissance, tree_penalty=tree_penalty),P(x_serliana, l_Baroque, tree_penalty=tree_penalty))))

    index_x_vano_adintelado = data[labels[:,14]==1]        
    if len(index_x_vano_adintelado):            
        x_vano_adintelado = ltn.Variable("x_vano_adintelado", index_x_vano_adintelado) 
        axioms.append(ForallP1(x_vano_adintelado,Or(P(x_vano_adintelado, l_Baroque, tree_penalty=tree_penalty),P(x_vano_adintelado, l_Renaissance, tree_penalty=tree_penalty))))
    ############################################
    return axioms 


# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader,tree_penalty,Predicate):
    mean_sat = 0
    for data, labels in loader:        
        axioms_classes = get_axioms_classes(data,labels,tree_penalty,Predicate)
        axioms_attr = get_axioms_attr(data,labels,tree_penalty,Predicate)
        axioms = axioms_classes+axioms_attr
        mean_sat += SatAgg(*axioms)
    mean_sat /= len(loader)
    return mean_sat

# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader,model):
    mean_accuracy = 0.0
    for data, labels in loader:
        predictions = model(data).detach().numpy()
        predictions = np.argmax(predictions, axis=1)
        mean_accuracy += accuracy_score(labels, predictions)

    return mean_accuracy / len(loader)