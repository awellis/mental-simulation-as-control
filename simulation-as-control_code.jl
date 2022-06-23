### A Pluto.jl notebook ###
# v0.19.8

#> [frontmatter]
#> author = "Andrew Ellis"
#> title = "Mental simulation as control"
#> date = "2022-06-16"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1b9aeb61-3534-4363-8e27-e7c9df717480
using Distributions, PlutoUI, DataFrames

# ╔═╡ 67d453be-1c76-4f2d-8660-76e5339f0953
using AlgebraOfGraphics, CairoMakie

# ╔═╡ dfb8573e-0617-4636-b16b-7f1e0270b0bd
using Turing

# ╔═╡ 45d7f55a-35de-47f6-9ce5-67fbbdeabaf3
using Base: @kwdef

# ╔═╡ fde7f730-eca3-11ec-11fd-a3669efbfb62
md"# Simulation as a control problem"

# ╔═╡ 9e4cf0fc-6654-4123-993e-39aa714b3731
md"""
It is commonly assumed that the brain's motor system makes use of internal models of the musculoskeletal system and the environment in order to control motor behaviour.

We will briefly discuss ideas originating in [Wolpert & Kawato (1998)](https://www.sciencedirect.com/science/article/pii/S0893608098000665) and explore their application to the mental simulation of behaviour.

> Humans exhibit an enormous repertoire of motor behavior which enables us to interact with many different objects under a variety of different environments.

Motor behavior is formulated as a control problem. In general, the problem of control can be considered as the computational process of determining the input to some system we wish to control ir order to achieve some desired output.

In essence, we can define _control_ as the attempt to make dynamical systems behave in a desirable way.

As an example, consider lifting a cup to ones lips. The desired output at time $t$ is a particular acceleration of the hand (as judged by proprioceptive sensory feedback). The input to the system must be a motor command.
"""

# ╔═╡ 7b82b21f-dd19-4f6b-89a4-917f3c7ad770
md"""
Motor commands depend on many variables:
- (varying) state of the body (positions, velocities, orientation relative to gravity)
- (unvarying) masses, moments of inertia, etc.
- knowledge of interactions with outside world (e.g. geometry and properties of cup)

All these (and more) can be considered as the context $C$ within which movement takes place. Motor commands must be tailored to take into account the current context.
"""

# ╔═╡ 45cd5b3a-0a3d-4e80-8cc1-33b734016ad6
md"""
## Forward and inverse models

The notion of an internal model, a system which mimics the behavior of a natural process, has emerged as an important theoretical concept in motor control.

There are two varieties of internal model, forward and inverse models. 

Forward models capture the forward or causal relationship between inputs to the system, e.g. the arm, and the outputs (Ito, 1970; Kawato et al., 1987; Jordan, 1995). A forward dynamic model of the arm, for example predicts the next state (e.g. position and velocity) given the current state and motor command.

In contrast, inverse models invert the system by providing the motor command which will cause a desired change in state. Inverse models are, therefore, well suited to act as controllers as they can provide the motor command necessary to achieve some desired state transition.

As both forward and inverse models depend on the dynamics of the motor system, which _change throughout life and under different contextual conditions, these models must be adaptable_.

## Multiple models

Wolpert & Kawato (1998) claim that there is evidence for the existence of multiple (independent) controllers (as opposed to one monolithic controller). The problem of choosing which model is responsible for actually controlling behaviour can be seen as a model selection problem.

__Assumptions__: At any given time, only certain controllers are active. Only the active controllers can be assessed for performance errors. 

The basic idea of Wolpert & Kawato's model is that multiple inverse models exist to control the system and each is augmented with a corresponding forward model. The brain therefore contains multiple pairs of coupled forward and inverse models. Further, there exists a _responsibility signal_ (for a given model) which reflects, at any given time, the degree to which each pair of forward and inverse models should be responsible for controlling the current behavior. This responsibility signal is derived from the combination of two processes. The first process uses the forward model’s predictions errors. The second process, the responsibility predictors, use sensory contextual cues to predict the responsibility of the module and can therefore select controllers prior to movement initiation in a feedforward manner.

> I don't know yet if it's worth exploring their model in great detail.




## Forward and inverse models as inference

In a more modern take, forward and inverse models can be interpreted as two types of inference in a probabilistic model. 

This will allow us to deal with uncertainty.
"""

# ╔═╡ 96f3de55-017f-4497-a9ee-0842c1e4a600
md"""
## Model

We consider an agent's motor system, which we wish to control by issuing a sequence of motor commands $u_{1...T}$. For simplicity, we consider discrete time. 

The resulting movement trajectory (state of the system) is $x_{1...T}$. 


The causal relationship between motor command $u$ and state $x$ is given by:

$x_{t+1} = f(x_t, u_t, c_t)$

This describes the _forward_ dynamics of the system. We assume that the dynamics of the system $f$ are not fixed over time but can take on a possibly infinite number of different forms. These different forms correspond to the context of the movement and include such factors as interactions with objects or changes in the environment. This can either be parameterized by assuming there is a set of system dynamics $f_i$ with $i = 1, 2, ..., n$ or by including a context parameter $c$ as part of the dynamics.

The aim of control is to produce a system which can generate an appropriate motor command $u_t$ given the desired state, $x_{t+1}^\ast$.
"""

# ╔═╡ 54513b17-f40a-4948-abfa-ff13e5fdc48b
md"""
## Simulations
"""

# ╔═╡ 147b95f9-903e-440e-92a4-f59a9f5897e8
md"""
Let's consider a simple agent that can move (change its position) along a unidimensional axis. We will denote the position at time $t$ by $θ_t$.
"""

# ╔═╡ bd29c3c5-8678-493c-b7ca-1209461d9941
PlutoUI.LocalResource("assets/bear-with-phone.png")

# ╔═╡ f29282a5-8f30-4759-9481-cf1153b91262
md"""
The agent is equipped with a GPS device, with which it can measure its position along the axis. However, the measurements are noisy, such that the observation a time $t$ is given by:

$$y_t \sim \mathcal{N}(\theta_{t}, \sigma^2_y)$$
"""

# ╔═╡ 93b6a338-4222-4f1c-a61f-4ae3c53a3569
md"""
The agent can move (left/right) by applying a force. Neglecting pretty much everything else (mass, friction, etc.), the applied force results in a single cycle sinusiodal acceleration. 

Therefore, the agent's motor action will result in an acceleration. We know that acceleration, $\alpha$, is the second derivative of position. We can therefore numerically integrate (twice) the acceleration over a time period to obtain the velocity, $\omega$, and then the position.

$$\begin{align}
\omega_t &=  \dot{θ}_t\\
\alpha_t &= \dot{\omega}_t = \ddot{\theta}_t = u_t\\
r_{t+1} &= r_t + c \, i_t
\end{align}$$
"""

# ╔═╡ 51799b48-ab66-49f6-b675-11d2f556f1b2
md"""

Using this information, we can write the agent's motion as a system of 1st order difference equations. These implement Newtonian kinematics.

$$\begin{align}
 u_t &= \alpha_t \\
\alpha_t &= D \cdot A  \cdot sin(2 * π * f * (t-\phi)) \\
\omega_{t+1} &= \omega_{t} + \Delta t \cdot \alpha_t \\
\theta_{t+1} &= \theta_{t} + \Delta t \cdot \omega_{t} + \frac{1}{2} \Delta t ^2 \cdot \alpha_t \\

\end{align}$$

where $D$, $A$, $\phi$ are the direction (sign), amplitude and phase, i.e. parameters of the sinusiodal acceleration. 
"""

# ╔═╡ db19811e-6680-4c7e-98d8-652b4d35214a
md"""
Let's try this out using a code example. You can use the sliders to set the amplitude, direction, onset and duration of motion.
"""

# ╔═╡ 6d343fb0-0eb5-43e7-ae6b-34c62a7da084
amplitude = @bind amplitude PlutoUI.Slider(0:20, default = 10)

# ╔═╡ 50159681-0c1c-4961-8318-d84ab7e95fce
direction = @bind direction Select(["left", "right"])

# ╔═╡ 43165e68-2265-4bf9-8f19-0dea04f29cfb
duration = @bind duration PlutoUI.Slider(0.5:6, default = 2)

# ╔═╡ f6fc7de8-db7b-4861-b045-b385f2ad97d5
onset = @bind onset PlutoUI.Slider(0:2, default = 1)

# ╔═╡ 46a0daaa-ace4-4ecb-8ff1-09a88b634a1c
md"""
Now we can simulate the result of applying an acceleration with the parameters defined above, and plot the resulting velocity, position and sensory observations.
"""

# ╔═╡ 2983a87e-7c52-447a-bbeb-3baeed09e611
show_observations = @bind show_observations CheckBox()

# ╔═╡ a7e2c25e-7b06-41a9-903f-65396ca1c09e
md"""
## Control 

What we did above was use the model to perform a forward simulation of the expected consequences of applying a motor command. The problem of controlling, i.e. selecting an appropriate action, is the inverse problem: in this case, figuring out which parameter settings to use in order to achieve a desired target position.

Say, for example, that the agent wishes to go to position $\theta = 10$. For simplicity, let's assume that the agent doens't care how long it takes to get there.


The problem can be construed in serveral ways. I propose that we either look at it as an optimization problem, or as a problem of Bayesian inference.
"""

# ╔═╡ 37079a26-df04-4de9-b2b2-4f2504a21ba7
md"""
### Control as optimization

"""

# ╔═╡ c6b44155-bd04-45dd-bba6-9b85a79d2487
md"""tbd..."""

# ╔═╡ ec74d6c2-e223-41cd-b1f7-3e6a6945d829
md"""
### Control as inference

The goal is to infer a posterior distribution over the parameters, conditioned upon setting $\theta = 10$. The parameters of interest are the duration and apmplitude; the direction of motion is given by the target position, since we are not (yet) considering circular motion.

The agent has to perform the following steps:
1) Set up probabilistic model.
2) Infer parameters, conditioned upon setting position to the desired value.
"""

# ╔═╡ 73a771b4-ad0f-4c64-bfb7-38aba7dc90f2
md"""
To be continued...
"""

# ╔═╡ b1161fc7-7779-44ec-a570-b335b3ba9745
md"""
## Code

"""

# ╔═╡ 10e1593e-0216-45e6-9c36-206e54848f67
begin

	@kwdef struct Sensor
    noise::Distribution = Normal(0, 1.0)
	end

	@kwdef struct PlannedMovement
    A::Real = 20 # amplitude
    D::String = "right" # direction
    onset::Real = 1
    duration::Real = 1
	end
	
	function acceleration(D::Number, A::Number, 
		f::Number, t::Number, start::Number)
		D * A * sin.(2 * π * f * (t-start))
	end
	
end

# ╔═╡ 57bea528-1106-4935-bfd3-c96426b39e15
mu = PlannedMovement(A=amplitude, D=direction, onset=onset, duration=duration);

# ╔═╡ 88b40a8d-900b-40c2-be5d-5fb32923919b
mu

# ╔═╡ 9ea6db4d-8859-4bb3-86eb-b16d91178dec
sensor = Sensor(noise=Normal(0, 1.0));

# ╔═╡ 832c29eb-49b1-41d4-98e7-960fb30ed9a1
begin
	function simulate(mᵤ::PlannedMovement, sensor::Sensor; Δt = 0.01, duration::Real = 2)
		
    onsetᵤ = mᵤ.onset
    @assert onsetᵤ >= 0
    @assert duration > 0

    motion_duration = mᵤ.duration
    total_duration = duration

    onsetᵤ +  motion_duration <= total_duration || error("Head turn extends beyond simulation event.")

    mᵤ.D ∈ ["left", "right"] || error("Direction not specified.")

    D = mᵤ.D == "left" ? -1 : 1
    A =mᵤ.A 

    noise = sensor.noise
    f = 1/(motion_duration) # frequency: single cycle sinusoidal 

    timesteps = range(0, stop = total_duration, step = Δt)
    T = length(timesteps)

    α = zeros(T)
    ω = zeros(T)
    θ = zeros(T)
    y = zeros(T)

    for i ∈ Iterators.drop(1:T, 1)
        t = round(timesteps[i]; digits = 3)
        α[i] = (t > onsetᵤ) & (t < onsetᵤ + motion_duration) ? acceleration(D, A, f, t, onsetᵤ) : 0
        ω[i] = ω[i-1] + Δt * α[i]
        θ[i] = θ[i-1] + Δt * ω[i] + 1/2 * Δt^2 * α[i]
        y[i] = θ[i] + rand(noise)
    end
    
    out = (timesteps = collect(timesteps),
            y = y,
            α = α, 
            ω = ω, 
            θ = θ, 
            Δt = Δt, onsetᵤ = onsetᵤ,
            sensor = noise, 
            mᵤ = mᵤ)
    return out

end
end

# ╔═╡ bb7be262-3487-4130-ab9c-8d1d46c51c3f
s = simulate(mu, sensor, Δt=0.01, duration=onset+duration+2);

# ╔═╡ 60af8174-7968-458f-9028-c2b80697e60c
@model function infer_motion(y, N; u, σω, σθ, σy, Δt)
    ω = tzeros(Real, N)
	θ = tzeros(Real, N)

    ω[1] ~ Normal(0, σω)
	θ[1] ~ Normal(0, σθ)
    y[1] ~ Normal(θ[1], σy)

    for i ∈ 2:N
        ω[i] ~ Normal(ω[i-1] + Δt * u[i], σω)
        # θ[i] ~ Normal(θ[i-1] + Δt * ω[i-1] + 0.5 * Δt^2 * u[i], σθ)
		θ[i] = θ[i-1] + Δt * ω[i-1] + 0.5 * Δt^2 * u[i]
        y[i] ~ Normal(θ[i], σy)
    end
	return θ
end

# ╔═╡ 50fe5674-cbde-4443-b3d9-4cc42beb59dd
begin 
	N = length(s.y)
    D = s.mᵤ.D == "left" ? -1 : 1
	
	σω = 0.05
	σθ = 0.05
	
    timesteps = range(0, stop=N * s.Δt, step=s.Δt)
    f = 1 / (duration)
    u = zeros(N)
    for i ∈ 2:N
        t = timesteps[i]
        u[i] = (t >= s.mᵤ.onset) && (t <= s.mᵤ.onset + s.mᵤ.duration) ? acceleration(D, s.mᵤ.A, f, t, s.mᵤ.onset) : 0.0
    end
	
	forward_model = infer_motion(s.y, N, u=u, σω=σω, σθ=σθ, σy=s.sensor.σ, Δt=s.Δt)	
	posterior = sample(forward_model, SMC(), 1000);
end

# ╔═╡ f7a342a5-b19e-456f-9d61-cc372fc1c51b
begin
    # chains_params = Turing.MCMCChains.get_sections(posterior, :θ)
    gen = generated_quantities(forward_model, posterior)
end

# ╔═╡ a3dcdc7d-4298-458f-ab7c-5a3ee55dd7fd
begin
	using Query
	d = DataFrame(group(posterior, :θ))
	select!(d, Not(:chain))
	colnames = vcat(:iteration, ["$i" for i in 1:ncol(d)-1])
	rename!(d, Symbol.(colnames))

	nsamples = 100
	dd = d[sample(axes(d, 1), nsamples; replace=false, ordered=true), :]

	dd = DataFrames.stack(dd, Not([:iteration]),
    	variable_name=:time, value_name=:θ)

	dd = dd |>
    @mutate(time = parse(Int64, _.time),
            iteration = string.(_.iteration)) |>
    @mutate(time = _.time*s.Δt) |> DataFrame
end

# ╔═╡ bb226991-1eec-44f1-a4eb-dbcd9126e6e7
@model function inferparams(y, θ_desired, N; u, σω, σθ, σy, Δt)
    ω = tzeros(Real, N)
	θ = tzeros(Real, N)

    ω[1] ~ Normal(0, σω)
	θ[1] ~ Normal(0, σθ)
    y[1] ~ Normal(θ[1], σy)

    for i ∈ 2:N
        ω[i] ~ Normal(ω[i-1] + Δt * u[i], σω)
        θ[i] = Normal(θ[i-1] + Δt * ω[i-1] + 0.5 * Δt^2 * u[i], σθ)
        y[i] ~ Normal(θ[i], σy)
    end
end

# ╔═╡ 324042ac-79cb-46a0-af2f-8db16e6bf717
begin
	function get_estimates(chain::Chains, var::Symbol)
    	x = Array(group(chain, var))
    	return (var=x)
	end
end

# ╔═╡ 3c0cc555-3670-4ce5-a6b8-bea14cbc06fc
θ = get_estimates(posterior, :θ);

# ╔═╡ df45ae13-8625-40fc-9520-322a47eb4855
ω = get_estimates(posterior, :ω);

# ╔═╡ 17d59070-ab0e-4c96-97d0-26a000723ec9
begin
	q025(x) = quantile(x, 0.025)
	q975(x) = quantile(x, 0.97)
	
	function compute_stats(x::Array)
    mean_x = mean.(eachslice(x, dims=2))
    xlb = mean_x .- q025.(eachslice(x, dims=2))
    xub = mean_x .- q975.(eachslice(x, dims=2))
    return (μ=mean_x, lb=xlb, ub=xub)
	end
end

# ╔═╡ 2bf1a8ef-16f0-4da7-9766-e58455df7e83
begin
	plt = data(dd) *
		visual(Lines, color = :steelblue3, linewidth = 0.2, alpha = 0.1) *
     	mapping(:time, :θ, group=:iteration)
	
	fig2 = draw(plt)
	
	μθ, lbθ, ubθ = compute_stats(θ)
	lines!(s.timesteps, μθ, 
        linewidth=2,
        linestyle = :dash,
        color=:steelblue3)
	current_figure()
end

# ╔═╡ 9a641cd1-f031-4dc9-837f-e9e69a13566d
md"""

## References

Wolpert, D. M., & Kawato, M. (1998). Multiple paired forward and inverse models for motor control. Neural Networks, 11(7), 1317–1329. [https://doi.org/10.1016/S0893-6080(98)00066-5](https://www.sciencedirect.com/science/article/pii/S0893608098000665)

"""

# ╔═╡ 2c12b0de-ed33-44b0-88d8-3f3195eb9e98
html"""<style>
main {
    max-width: 100%;
}
"""

# ╔═╡ 590e843d-5881-4e41-aab8-3c4ef6bc83bc
TableOfContents(title="📚 Table of Contents", 
	indent=true, depth=4, aside=true)

# ╔═╡ 2ee44f38-df4c-4109-b099-eb159fc0d0fa
begin
	publication_theme() = Theme(
		fontsize=18, font="sans",
		Axis=(xlabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ytickalign=1, yticksize=10, xticksize=10,
        xlabelpadding=-5, ylim=(0, 5)), 
		Legend=(framecolor=(:black, 0.5), 
		bgcolor=(:white, 0.5)),
		Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5))
end

# ╔═╡ b0adc44e-8e3f-4f9e-b183-c943fd122c69
begin
	with_theme(publication_theme()) do
    f1 = lines(s.timesteps, s.α,
        linewidth=3,
        color=:black,
		linestyle=:solid,
		label = "Acceleration",
        axis=(xticks=LinearTicks(6),
            xlabel="Time (s)",
            ylabel="Position (arbitrary units)",
            xgridstyle=:dash, ygridstyle=:dash))
	f2 = lines!(s.timesteps, s.ω,
        linewidth=6,
        color=:darkorange,
	label = "Velocity")	
	f3 = lines!(s.timesteps, s.θ,
        linewidth=6,
        color=:steelblue3,
	    linestyle=:solid,
	label = "Position")	
	f4 = show_observations && scatter!(s.timesteps, s.y,
	        color=(:steelblue3),
			marker='◆',
        	markersize=6,
	label = "Observations")
	vspan!([0, mu.onset + mu.duration], [mu.onset, maximum(s.timesteps)],color = [(c, 0.2) for c in [:grey, :grey]])
    ylims!(min(minimum(s.α), minimum(s.θ)) - 1, max(maximum(s.α), maximum(s.θ)) + 1)
	axislegend(position = :rb, orientation = :horizontal)
    current_figure()
end
end

# ╔═╡ 02d3cd9d-3c53-49e2-882c-056f3225bb27
begin
	fig1 = with_theme(publication_theme()) do
    μθ, lbθ, ubθ = compute_stats(θ)
    μω, lbω, ubω = compute_stats(ω)

    lines(s.timesteps, μω,
        linewidth=4,
        color="darkorange",
        axis=(xticks=LinearTicks(6),
            xlabel="Time (s)",
            ylabel="Inferred position/velocity",
            xgridstyle=:dash, ygridstyle=:dash))
    lines!(s.timesteps, μθ,
        linewidth=4,
        color=:steelblue3)
    band!(s.timesteps, μω - lbω, μω - ubω, color=(:darkorange, 0.5))
    band!(s.timesteps, μθ - lbθ, μθ - ubθ, color=(:steelblue3, 0.5))
    lines!(s.timesteps, s.θ, linewidth=4, linestyle=:dash, color=:black)
    current_figure()
end

end

# ╔═╡ a83e98d0-1275-437e-947d-d14613432253
note(text) = Markdown.MD(Markdown.Admonition("note", "Note", [text]))

# ╔═╡ d9f6f38e-0a10-411e-abf7-64fc75277963
note(md"""
The agent's observations are of course a model of a sensory system. We are assuming that the sensor noise is homoscedastic.""")

# ╔═╡ d62199a4-c460-484f-9793-b49af77b15bb
note(md"""
We are assuming that the agent can only perform one type of motion (sinusiodal acceleration), and has perfect knowledge of the form of the acceleration. Further, the acceleration is parameterized using $D$, $A$, $\phi$. $f$ is set to $1$ because the agent's motor system can only perform a single cycle pf motion.
""")

# ╔═╡ Cell order:
# ╠═1b9aeb61-3534-4363-8e27-e7c9df717480
# ╠═67d453be-1c76-4f2d-8660-76e5339f0953
# ╠═dfb8573e-0617-4636-b16b-7f1e0270b0bd
# ╠═45d7f55a-35de-47f6-9ce5-67fbbdeabaf3
# ╟─fde7f730-eca3-11ec-11fd-a3669efbfb62
# ╟─9e4cf0fc-6654-4123-993e-39aa714b3731
# ╟─7b82b21f-dd19-4f6b-89a4-917f3c7ad770
# ╟─45cd5b3a-0a3d-4e80-8cc1-33b734016ad6
# ╟─96f3de55-017f-4497-a9ee-0842c1e4a600
# ╟─54513b17-f40a-4948-abfa-ff13e5fdc48b
# ╟─147b95f9-903e-440e-92a4-f59a9f5897e8
# ╟─bd29c3c5-8678-493c-b7ca-1209461d9941
# ╟─f29282a5-8f30-4759-9481-cf1153b91262
# ╟─d9f6f38e-0a10-411e-abf7-64fc75277963
# ╟─93b6a338-4222-4f1c-a61f-4ae3c53a3569
# ╟─51799b48-ab66-49f6-b675-11d2f556f1b2
# ╟─d62199a4-c460-484f-9793-b49af77b15bb
# ╟─db19811e-6680-4c7e-98d8-652b4d35214a
# ╠═6d343fb0-0eb5-43e7-ae6b-34c62a7da084
# ╟─50159681-0c1c-4961-8318-d84ab7e95fce
# ╟─43165e68-2265-4bf9-8f19-0dea04f29cfb
# ╟─f6fc7de8-db7b-4861-b045-b385f2ad97d5
# ╟─46a0daaa-ace4-4ecb-8ff1-09a88b634a1c
# ╠═57bea528-1106-4935-bfd3-c96426b39e15
# ╠═9ea6db4d-8859-4bb3-86eb-b16d91178dec
# ╠═bb7be262-3487-4130-ab9c-8d1d46c51c3f
# ╟─2983a87e-7c52-447a-bbeb-3baeed09e611
# ╟─b0adc44e-8e3f-4f9e-b183-c943fd122c69
# ╟─a7e2c25e-7b06-41a9-903f-65396ca1c09e
# ╟─37079a26-df04-4de9-b2b2-4f2504a21ba7
# ╟─c6b44155-bd04-45dd-bba6-9b85a79d2487
# ╟─ec74d6c2-e223-41cd-b1f7-3e6a6945d829
# ╟─73a771b4-ad0f-4c64-bfb7-38aba7dc90f2
# ╠═88b40a8d-900b-40c2-be5d-5fb32923919b
# ╠═50fe5674-cbde-4443-b3d9-4cc42beb59dd
# ╠═3c0cc555-3670-4ce5-a6b8-bea14cbc06fc
# ╠═df45ae13-8625-40fc-9520-322a47eb4855
# ╠═f7a342a5-b19e-456f-9d61-cc372fc1c51b
# ╠═02d3cd9d-3c53-49e2-882c-056f3225bb27
# ╠═a3dcdc7d-4298-458f-ab7c-5a3ee55dd7fd
# ╠═2bf1a8ef-16f0-4da7-9766-e58455df7e83
# ╟─b1161fc7-7779-44ec-a570-b335b3ba9745
# ╠═10e1593e-0216-45e6-9c36-206e54848f67
# ╠═832c29eb-49b1-41d4-98e7-960fb30ed9a1
# ╠═60af8174-7968-458f-9028-c2b80697e60c
# ╠═bb226991-1eec-44f1-a4eb-dbcd9126e6e7
# ╠═324042ac-79cb-46a0-af2f-8db16e6bf717
# ╠═17d59070-ab0e-4c96-97d0-26a000723ec9
# ╟─9a641cd1-f031-4dc9-837f-e9e69a13566d
# ╠═2c12b0de-ed33-44b0-88d8-3f3195eb9e98
# ╠═590e843d-5881-4e41-aab8-3c4ef6bc83bc
# ╠═2ee44f38-df4c-4109-b099-eb159fc0d0fa
# ╠═a83e98d0-1275-437e-947d-d14613432253
