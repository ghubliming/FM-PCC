import os

path = '/workspaces/FM-PCC/logs_in_develop/HF_iMF/HF_Study/MAP_Algorithm1_to_AvoidingCode.md'
with open(path, 'r') as f:
    text = f.read()

# Section 1
text = text.replace('`(α_t, β_t)`', '$(\\alpha_t, \\beta_t)$')
text = text.replace('`X_t = α_t·X_1 + β_t·X_0`', '$X_t = \\alpha_t X_1 + \\beta_t X_0$')
text = text.replace('`α_t = t`, `β_t = 1-t`', '$\\alpha_t = t$, $\\beta_t = 1-t$')
text = text.replace('`α̇_t`, `β̇_t`', '$\\dot{\\alpha}_t$, $\\dot{\\beta}_t$')
text = text.replace('`Λ_t := α_t β̇_t - α̇_t β_t`', '$\\Lambda_t := \\alpha_t \\dot{\\beta}_t - \\dot{\\alpha}_t \\beta_t$')
text = text.replace('`t·(-1) - 1·(1-t) = -1`', '$t(-1) - 1(1-t) = -1$')
text = text.replace('`v_{t\\mid Z} = α̇_t X_1 + β̇_t X_0`', '$v_{t\\mid Z} = \\dot{\\alpha}_t X_1 + \\dot{\\beta}_t X_0$')
text = text.replace('`1·X_1 + (-1)·X_0`', '$1\\cdot X_1 + (-1)\\cdot X_0$')

# Section 2
text = text.replace('`v_t^θ(x)`', '$v_t^\\theta(x)$')
text = text.replace('`ℝ^d`', '$\\mathbb{R}^d$')
text = text.replace('`x_i`, `\\bar{x}_{i+1}`, `x_{i+1}`', '$x_i$, $\\bar{x}_{i+1}$, $x_{i+1}$')
text = text.replace('`t_i`, `Δt_i`', '$t_i$, $\\Delta t_i$')
text = text.replace('`N` (discretization steps)', '$N$ (discretization steps)')
text = text.replace('`𝓜_t^θ(x) := E[X_1∣X_t=x]`', '$\\mathcal{M}_t^\\theta(x) := \\mathbb{E}[X_1 \\mid X_t=x]$')
text = text.replace('`\\bar{x}_N := 𝓜_{t_{i+1}}^θ(\\bar{x}_{i+1})`', '$\\bar{x}_N := \\mathcal{M}_{t_{i+1}}^\\theta(\\bar{x}_{i+1})$')
text = text.replace('`\\hat{x}_N^*`', '$\\hat{x}_N^*$')
text = text.replace('`h(x) ≤ 0`', '$h(x) \\le 0$')
text = text.replace('`C(x)`', '$C(x)$')
text = text.replace('`λ_oc`', '$\\lambda_{oc}$')
text = text.replace('`α_{t_{i+1}}`', '$\\alpha_{t_{i+1}}$')
text = text.replace('`λ_oc/(2Δt_i)·α_{t_{i+1}}²`', '$\\frac{\\lambda_{oc}}{2\\Delta t_i} \\alpha_{t_{i+1}}^2$')
text = text.replace('`u_i^*`', '$u_i^*$')
text = text.replace('`\\bar{x}_N`', '$\\bar{x}_N$')
text = text.replace('`t_{i+1}`', '$t_{i+1}$')
text = text.replace('`Λ_t`', '$\\Lambda_t$')

# Section 3
text = text.replace('Inputs: `p_0`, `v_t^θ`, `C(·)`, `h(·)≤0`, `λ_oc`, `N`, time grid, scheduler `(α_t,β_t)`.', 'Inputs: $p_0$, $v_t^\\theta$, $C(\\cdot)$, $h(\\cdot)\\le 0$, $\\lambda_{oc}$, $N$, time grid, scheduler $(\\alpha_t, \\beta_t)$.')

text = text.replace('### Line: `Draw initial state \\bar{x}_0 ~ p_0 and set x_0=\\bar{x}_0`', '### Line: `Draw initial state` $\\bar{x}_0 \\sim p_0$ `and set` $x_0=\\bar{x}_0$')
text = text.replace('`x_0`', '$x_0$')

text = text.replace('### Line: `for i = 0 to N-1:`', '### Line: `for` $i = 0$ `to` $N-1$`:')
text = text.replace('### Line: `Compute Δt_i = t_{i+1} - t_i`', '### Line: `Compute` $\\Delta t_i = t_{i+1} - t_i$')

text = text.replace('`Δt_i`', '$\\Delta t_i$')

text = text.replace('### Line: `Compute \\bar{x}_{i+1} = x_i + v_{t_i}^θ(x_i)·Δt_i`', '### Line: `Compute` $\\bar{x}_{i+1} = x_i + v_{t_i}^\\theta(x_i) \\Delta t_i$')

text = text.replace('### Line: `Compute \\bar{x}_N = (β̇_{t_{i+1}}·\\bar{x}_{i+1} - β_{t_{i+1}}·v_{t_{i+1}}^θ(\\bar{x}_{i+1})) / Λ_{t_{i+1}}`', '### Line: `Compute` $\\bar{x}_N = \\frac{\\dot{\\beta}_{t_{i+1}} \\bar{x}_{i+1} - \\beta_{t_{i+1}} v_{t_{i+1}}^\\theta(\\bar{x}_{i+1})}{\\Lambda_{t_{i+1}}}$')

text = text.replace('`𝓜_{t_{i+1}}^θ(\\bar{x}_{i+1})`', '$\\mathcal{M}_{t_{i+1}}^\\theta(\\bar{x}_{i+1})$')
text = text.replace('`β̇=-1, β=1-t, Λ=-1`', '$\\dot{\\beta}=-1, \\beta=1-t, \\Lambda=-1$')

text = text.replace('```\n\\bar{x}_N = ((-1)·\\bar{x}_{i+1} - (1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})) / (-1)\n          = \\bar{x}_{i+1} + (1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})\n```', '$$\n\\begin{aligned}\n\\bar{x}_N &= \\frac{(-1) \\bar{x}_{i+1} - (1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1})}{-1} \\\\\n          &= \\bar{x}_{i+1} + (1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1})\n\\end{aligned}\n$$')

text = text.replace('```\n\hat{x}_N^* = argmin_{\hat{x}_N}  C(\hat{x}_N) + λ_oc/(2Δt_i)·α_{t_{i+1}}²·‖\hat{x}_N - \\bar{x}_N‖²\n              s.t. h(\hat{x}_N) ≤ 0\n```', '$$\n\\begin{aligned}\n\\hat{x}_N^* &= \\arg\\min_{\\hat{x}_N} C(\\hat{x}_N) + \\frac{\\lambda_{oc}}{2\\Delta t_i} \\alpha_{t_{i+1}}^2 \\|\\hat{x}_N - \\bar{x}_N\\|^2 \\\\\n&\\text{s.t.} \\quad h(\\hat{x}_N) \\le 0\n\\end{aligned}\n$$')

text = text.replace('`λ_oc/Δt_i`', '$\\lambda_{oc}/\\Delta t_i$')
text = text.replace('`1/Δt_i`', '$1/\\Delta t_i$')
text = text.replace('`=t_{i+1}²`', '$=t_{i+1}^2$')
text = text.replace('`C(\\hat{x}_N)`', '$C(\\hat{x}_N)$')
text = text.replace('`h(\\hat{x}_N)≤0`', '$h(\\hat{x}_N) \\le 0$')

text = text.replace('### Line: `Compute x_{i+1} = α_{t_{i+1}}·\\hat{x}_N^* + β_{t_{i+1}}·(-α̇_{t_{i+1}}·\\bar{x}_{i+1}+α_{t_{i+1}}·v_{t_{i+1}}^θ(\\bar{x}_{i+1})) / Λ_{t_{i+1}}`', '### Line: `Compute` $x_{i+1} = \\alpha_{t_{i+1}} \\hat{x}_N^* + \\beta_{t_{i+1}} \\frac{-\\dot{\\alpha}_{t_{i+1}} \\bar{x}_{i+1} + \\alpha_{t_{i+1}} v_{t_{i+1}}^\\theta(\\bar{x}_{i+1})}{\\Lambda_{t_{i+1}}}$')

text = text.replace('`α_{t_{i+1}}·\\hat{x}_N^* + β_{t_{i+1}}·𝒲_{t_{i+1}}^θ(\\bar{x}_{i+1})`', '$\\alpha_{t_{i+1}} \\hat{x}_N^* + \\beta_{t_{i+1}} \\mathcal{W}_{t_{i+1}}^\\theta(\\bar{x}_{i+1})$')
text = text.replace('`𝒲`', '$\\mathcal{W}$')

text = text.replace('`α=t, β=1-t, α̇=1, Λ=-1`', '$\\alpha=t, \\beta=1-t, \\dot{\\alpha}=1, \\Lambda=-1$')

text = text.replace('```\nx_{i+1} = t_{i+1}·\\hat{x}_N^* + (1-t_{i+1})·(-\\bar{x}_{i+1} + t_{i+1}·v_{t_{i+1}}(\\bar{x}_{i+1})) / (-1)\n        = t_{i+1}·\\hat{x}_N^* + (1-t_{i+1})·(\\bar{x}_{i+1} - t_{i+1}·v_{t_{i+1}}(\\bar{x}_{i+1}))\n        = (1-t_{i+1})·\\bar{x}_{i+1} + t_{i+1}·\\hat{x}_N^* - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})      (*)\n```', '$$\n\\begin{aligned}\nx_{i+1} &= t_{i+1} \\hat{x}_N^* + (1-t_{i+1}) \\frac{-\\bar{x}_{i+1} + t_{i+1} v_{t_{i+1}}(\\bar{x}_{i+1})}{-1} \\\\\n        &= t_{i+1} \\hat{x}_N^* + (1-t_{i+1}) (\\bar{x}_{i+1} - t_{i+1} v_{t_{i+1}}(\\bar{x}_{i+1})) \\\\\n        &= (1-t_{i+1}) \\bar{x}_{i+1} + t_{i+1} \\hat{x}_N^* - t_{i+1}(1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1}) \\quad (*)\n\\end{aligned}\n$$')

text = text.replace('`x_{i+1} = \\bar{x}_{i+1} + t_{i+1}·(\\hat{x}_N^* - \\bar{x}_N)`', '$x_{i+1} = \\bar{x}_{i+1} + t_{i+1}(\\hat{x}_N^* - \\bar{x}_N)$')
text = text.replace('`\\bar{x}_N = \\bar{x}_{i+1} + (1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})`', '$\\bar{x}_N = \\bar{x}_{i+1} + (1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1})$')

text = text.replace('```\nx_{i+1} = \\bar{x}_{i+1} + t_{i+1}·\\hat{x}_N^* - t_{i+1}·\\bar{x}_{i+1} - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})\n        = (1-t_{i+1})·\\bar{x}_{i+1} + t_{i+1}·\\hat{x}_N^* - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\\bar{x}_{i+1})\n```', '$$\n\\begin{aligned}\nx_{i+1} &= \\bar{x}_{i+1} + t_{i+1} \\hat{x}_N^* - t_{i+1} \\bar{x}_{i+1} - t_{i+1}(1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1}) \\\\\n        &= (1-t_{i+1}) \\bar{x}_{i+1} + t_{i+1} \\hat{x}_N^* - t_{i+1}(1-t_{i+1}) v_{t_{i+1}}(\\bar{x}_{i+1})\n\\end{aligned}\n$$')

text = text.replace('`x_next_ref + t_{i+1}·(x̂ - \\bar{x}_N)`', '$x_{next\\_ref} + t_{i+1}(\\hat{x}_N^* - \\bar{x}_N)$')
text = text.replace('`x = α_t·𝓜_t(x) + β_t·𝒲_t(x)`', '$x = \\alpha_t \\mathcal{M}_t(x) + \\beta_t \\mathcal{W}_t(x)$')

text = text.replace('### Line: `\\KwOut{Sample x_N}`', '### Line: `\\KwOut{Sample ` $x_N$ `}`')
text = text.replace('`x ∈ ℝ^d`', '$x \\in \\mathbb{R}^d$')
text = text.replace('`x = [a_0, s_1, a_1, s_2, …, a_{H-1}, s_H]`', '$x = [a_0, s_1, a_1, s_2, \\dots, a_{H-1}, s_H]$')
text = text.replace('`s_0`', '$s_0$')
text = text.replace('`v_t^θ(x)`', '$v_t^\\theta(x)$')
text = text.replace('`h(x) ≤ 0`', '$h(x) \\le 0$')

text = text.replace('`‖pos - pillar_center‖ ≥ radius + margin` ⟺ `h = radius+margin-‖pos-center‖ ≤ 0`', '$\\|pos - pillar\\_center\\| \\ge radius + margin \\iff h = radius+margin-\\|pos-center\\| \\le 0$')

text = text.replace('`A·s + B·a + c = s\'`', '$A s + B a + c = s\'$')
text = text.replace('`h(x_N)≤0`', '$h(x_N) \\le 0$')
text = text.replace('`C≠0`', '$C \\ne 0$')

text = text.replace('`x_N`', '$x_N$')
text = text.replace('`x_next_ref + 0·v_next = x_next_ref`', '$x_{next\\_ref} + 0\\cdot v_{next} = x_{next\\_ref}$')
text = text.replace('`1-t_{i+1}=1-1=0`', '$1-t_{i+1}=1-1=0$')
text = text.replace('`x_next = x_next_ref + 1·(x_terminal_predicted - x_terminal_predicted_ref)\n= x_terminal_predicted = \\hat{x}_N^*`', '$x_{next} = x_{next\\_ref} + 1\\cdot (x_{terminal\\_predicted} - x_{terminal\\_predicted\\_ref}) = x_{terminal\\_predicted} = \\hat{x}_N^*$')
text = text.replace('`x_next = x_next_ref + 1·(x_terminal_predicted - x_terminal_predicted_ref) = x_terminal_predicted = \\hat{x}_N^*`', '$x_{next} = x_{next\\_ref} + 1\\cdot (x_{terminal\\_predicted} - x_{terminal\\_predicted\\_ref}) = x_{terminal\\_predicted} = \\hat{x}_N^*$')

text = text.replace('`𝓜_{t_N}^θ(x_N)=x_N`', '$\\mathcal{M}_{t_N}^\\theta(x_N)=x_N$')

with open(path, 'w') as f:
    f.write(text)
