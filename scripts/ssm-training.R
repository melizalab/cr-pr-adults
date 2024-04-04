### run state-space model analysis on pretraining for a bird (supply as commandline arg)
set.seed(123)
n_iter <- 1e5

## load packages
import <- function(pkg) { library(pkg, warn.conflicts=F, quietly=T, character.only=T) }
import("stringr")
import("tidyr")
import("dplyr")
import("bssm")

## function definitions
logit <- function(p) log(p / (1 - p))
invlogit <- function(x) exp(x) / (1 + exp(x))

## parse commandline arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("usage: Rscript ssm-pretraining.R <dataset>\n")
}
dataset <- args[1]

header <- data.table::fread(cmd=str_c("ls build/", dataset, "*_trials.csv | head -n1 | xargs head -n1"), header=T)
all_trials <- tibble(data.table::fread(cmd=str_c("ls build/", dataset, "*_trials.csv | xargs tail -q -n+2"), header=F))
names(all_trials) <- names(header)

## load data. The trials are numbered before doing any exclusions so state can be back-referenced accordingly
trials <- (
    all_trials
    |> arrange(id)    # use this instead of time because there are clock errors for one subj
    |> mutate(noresp=(response == "timeout") * 1,
              # NB this is only valid for the trials when response is peck_left or peck_right
              stim_left=1 - xor(response=="peck_left", correct),
              peck_left=ifelse(noresp, NA, (response == "peck_left") * 1),
              correct=ifelse(noresp, NA, correct * 1),
              trial=row_number(),
              tot_rewarded=cumsum(result=="feed"),
              tot_noresp=cumsum(response=="timeout"),
              ## trials are considered to be corrections if the stimulus was repeated and the previous trial was incorrect
              inferred_correction=(lag(stimulus)==stimulus & lag(!correct)))
)
subject <- first(trials$subject)
cat("- Loaded", nrow(trials), "trials for subject", subject, "\n")

valid_trials <- (
    trials
    |> filter(str_length(lights)==0, !inferred_correction, response != "peck_center") 
    |> mutate(time=row_number())
)
trial_lookup <- select(valid_trials, trial, time)
cat("- Included", nrow(valid_trials), "valid trials\n")

cat("- Sampling posterior for p(nopeck)\n")
noresp_model <- bsm_ng(valid_trials$noresp, sd_level=halfnormal(0.1, 1), distribution="binomial", u=1)
noresp_samples <- run_mcmc(noresp_model, iter = n_iter, particles = 10)

## cat("- Sampling posterior for p(correct) with missing data\n")
## corr_model <- bsm_ng(valid_trials$correct, sd_level=halfnormal(0.1, 1), distribution="binomial", u=1, P1=0.01)
## corr_samples <- run_mcmc(corr_model, iter = n_iter, particles = 10)

cat("- Sampling posterior for p(left|stim) with missing data\n")
Z <- model.matrix(~ stim_left, valid_trials) 
Z[,2] <- Z[,2] - 0.5

# update R based on current value of theta
update_fn <- function(theta) {
    R <- diag(theta)
    dim(R) <- c(2, 2, 1)
    list(R=R)
}

# gamma prior: this has a lot of prior weight at zero and tails off around 2
prior_fn <- function(theta) {
    sum(dgamma(theta, 1, 2, log=TRUE))
}
resp_model <- ssm_ung(valid_trials$peck_left, t(Z), T=diag(2), R=diag(2), a1=c(0, 0), P1=diag(c(2,0.1)),
                      init_theta=c(sd_bias=0.1, sd_discrim=0.1),
                      update_fn=update_fn,
                      prior_fn=prior_fn,
                      distribution="binomial", 
                      state_names=c("bias", "discrim"))
resp_samples <- run_mcmc(resp_model, iter = n_iter, particles = 10)

results <- list(data=valid_trials,
		noresp_model=noresp_model,
		noresp_samples=noresp_samples,
		resp_model=resp_model,
		resp_samples=resp_samples)
saveRDS(results, file=str_c("build/", dataset, "_ssm.rds"))

cat("- Computing summary statistics for p(nopeck)\n")
noresp_samples$alpha <- invlogit(noresp_samples$alpha)
summary_noresp <- (
    summary(noresp_samples, variable="states", probs=c(0.05, 0.95))
    |> rename(mean=Mean, lwr="5%", upr="95%")
    |> inner_join(trial_lookup, by="time")
)    

## cat("- Computing summary statistics for p(correct)\n")
## corr_samples$alpha <- invlogit(corr_samples$alpha)
## summary_corr <- (
##     summary(corr_samples, variable="states", probs=c(0.05, 0.95))
##     |> rename(mean=Mean, lwr="5%", upr="95%")
##     |> inner_join(trial_lookup, by="time")
## )

cat("- Computing summary statistics for p(peck|stim)\n")
summary_discrim <- (
    summary(resp_samples, variable="states", probs=c(0.05, 0.95))
    |> rename(mean=Mean, lwr="5%", upr="95%")
    |> inner_join(trial_lookup, by="time")
)
samples <- rlang::duplicate(resp_samples)
samples$alpha[,1,] <- invlogit(resp_samples$alpha[,"bias",] + 0.5 * resp_samples$alpha[,"discrim",])
samples$alpha[,2,] <- invlogit(resp_samples$alpha[,"bias",] - 0.5 * resp_samples$alpha[,"discrim",])
colnames(samples$alpha) <- c("p_left", "p_right")
summary_peck <- (
    summary(samples, variable="states", probs=c(0.05, 0.95))
    |> rename(mean=Mean, lwr="5%", upr="95%")
    |> inner_join(trial_lookup, by="time")
)

saveRDS(list(data=valid_trials, noresp=summary_noresp, discrim=summary_discrim, peck=summary_peck),
        file=str_c("build/", dataset, "_ssm_summary.rds"))


