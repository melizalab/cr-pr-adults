### compute selectivity for each unit using generalized linear models
set.seed(123)

## load packages
import <- function(pkg) { library(pkg, warn.conflicts=F, quietly=T, character.only=T) }
import("stringr")
import("tidyr")
import("dplyr")
import("lme4")
import("emmeans")

## this doesn't really belong here, but we need to generate lists of CR and PR units for the decoder
birds <- data.table::fread("datasets/zebf-social-acoustical-ephys/metadata/birds.csv")
sites <- data.table::fread("datasets/zebf-social-acoustical-ephys/metadata/recordings.csv")
all_sites <- (
   sites
   |> mutate(bird=str_match(site, "[:alnum:]+")[,1])
   |> inner_join(birds, by="bird")
   |> mutate(group=factor(group, levels=c("CR", "PR")))
)
all_units <- (
    data.table::fread("datasets/zebf-social-acoustical-ephys/metadata/mean_spike_features.csv")
    |> mutate(spike=factor(spike, levels=c("wide", "narrow"), exclude=""))
    |> filter(!is.na(spike))
    |> mutate(site=str_match(unit, "[:alnum:]+_\\d+_\\d+")[,1])
    |> select(unit, site, spike)
    |> left_join(all_sites, by="site")
)
filter(all_units, group=="CR") |> select(unit) |> readr::write_csv("build/cr_units.txt", col_names=F)
filter(all_units, group=="PR") |> select(unit) |> readr::write_csv("build/pr_units.txt", col_names=F)

## load the rate data
header <- data.table::fread(cmd='find build/ -name "*_rates.csv" | head -n1 | xargs head -n1', header=T)
all_motif_rates <- tibble(data.table::fread(cmd='find build/ -name "*_rates.csv" | xargs tail -q -n+2', header=F))
names(all_motif_rates) <- names(header)

## initial cleaning - only responses to highest SNR, omit background segment
motif_rates <- (
    all_motif_rates
    |> filter(background_dBFS==-100 | foreground=="silence", foreground!="background")
    |> mutate(foreground=relevel(factor(foreground), "silence"))
)
## pool trials of the same stimulus (we can do this because our dependent variable is Poisson)
## and regularize rate estimates by adding 1 spike to units with no spontaneous spikes
motif_rate_summary <- (
    motif_rates
    |> group_by(unit, foreground) 
    |> summarize(n_events=sum(n_events), interval=sum(interval_end))
    |> mutate(n_events=ifelse(foreground=="silence" & n_events == 0, 1, n_events))
)

## fit the GLM to an individual unit
rate_model <- function(df) {
    glm(n_events ~ foreground, data=df, offset=log(interval), family=poisson)
}

## get the estimated rates using marginal means
rate_model_responsive <- function(mdl) {
    (
        emmeans(mdl, ~ foreground) 
        |> contrast("trt.vs.ctrl") 
        |> broom::tidy() 
        |> transmute(foreground=str_extract(contrast, "\\w+"), estimate, is_responsive=(estimate > 0) & (adj.p.value < 0.05))
    )
}

## run the model for all the units. This can take a while.
unit_motif_responsive <- (
    motif_rate_summary
    |> group_by(unit)
    |> nest()
    |> mutate(model=purrr::map(data, rate_model, .progress=TRUE))
    |> mutate(responsive=purrr::map(model, rate_model_responsive))
    |> select(unit, responsive)
    |> unnest(cols=responsive)
)

## summarize each unit's average evoked rate and the number of motifs that elicit a significant response (selectivity)
unit_rate_stats <- (
    unit_motif_responsive
    |> group_by(unit)
    |> summarize(avg_evoked=mean(estimate), n_responsive=sum(is_responsive))
)

readr::write_csv(unit_motif_responsive, "build/motif_rate_glm.csv")
readr::write_csv(unit_rate_stats, "build/unit_rate_stats.csv")