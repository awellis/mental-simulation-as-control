
using Distributions
using Base: @kwdef
using CairoMakie

@kwdef struct Sensor
    noise::Distribution = Normal(0, 1.0)
end

@kwdef struct PlannedHeadTurn
    A::Real = 20 # amplitude
    D::String = "right" # direction
    onset::Real = 1
    duration::Real = 1
    # @assert A >= zero(A)
end
	
function acceleration(D::Number, A::Number, 
    f::Number, t::Number, start::Number)
	D * A * sin.(2*π*f*(t-start))
end


function simulate(mᵤ::PlannedHeadTurn,
    sensor::Sensor;
    Δt = 0.01,
    duration::Real = 2)
		
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
    f = 1/(motion_duration) # frequency: single sinusoidal head turn

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
        y[i] = ω[i] + rand(noise)
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


amplitude, direction = 12, "right"

mu = PlannedHeadTurn(A=amplitude, D=direction, onset=0.1, duration=1.0);

sensor = Sensor(noise=Normal(0, 0.5));

s = simulate(mu, sensor, Δt=0.01, duration=2);


publication_theme() = Theme(
		fontsize=24, font="sans",
		Axis=(xlabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ytickalign=1, yticksize=10, xticksize=10,
        xlabelpadding=-5, ylim=(0, 5)), 
		Legend=(framecolor=(:black, 0.5), 
		bgcolor=(:white, 0.5)),
		Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5))

show_observations = true

f1 = with_theme(publication_theme()) do
    lines(s.timesteps, s.α,
        linewidth=3,
        color="black",
		linestyle=:dash,
        axis=(xticks=LinearTicks(6),
            xlabel="Time (s)",
            ylabel="Angular velocity (deg/s)",
            xgridstyle=:dash, ygridstyle=:dash))
	lines!(s.timesteps, s.ω,
        linewidth=6,
        color="darkorange")	
	lines!(s.timesteps, s.θ,
        linewidth=6,
        color="blue",
	    linestyle=:solid)	
	show_observations && scatter!(s.timesteps, s.y,
	        color=(:black),
			marker='◆',
        	markersize=6)
	vspan!([0, 0.3, 1], [0.2, 0.4, 1],
    color = [(c, 0.2) for c in [:red, :orange, :pink]])
    ylims!(minimum(s.α) - 1, maximum(s.α) + 1)
    current_figure()
end