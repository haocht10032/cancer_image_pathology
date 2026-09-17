# CPU-only reporting from frozen outputs. Does not modify experiment artifacts.
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})
arg <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
paper <- dirname(normalizePath(gsub("~+~", " ", arg, fixed=TRUE)))
root <- dirname(paper)
out <- file.path(root, "artifacts/feedback2_revision/reporting")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
fig <- file.path(paper, "manuscript_figures")
dir.create(fig, recursive=TRUE, showWarnings=FALSE)
read <- function(p) read.csv(file.path(root, p), stringsAsFactors=FALSE)
write <- function(d, name) write.csv(d, file.path(out, paste0(name, ".csv")), row.names=FALSE)
models <- c("ResNet18", "DINOv2", "UNI")
methods <- c(ResNet18="gradcam", DINOv2="gradient_attention_rollout", UNI="gradient_attention_rollout")
metrics <- c("attribution_occlusion_spearman", "top_beats_random_target_logit",
             "top_minus_random_relative_target_logit_reduction_auc")
short <- c("Spearman", "Top beats random", "Relative logit AUC (secondary)")
names(short) <- metrics
finite_mean <- function(x) if(any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
clean <- function(d) d %>% filter(model %in% models, method==unname(methods[model]),
    perturbation=="normalized_zero", grid_label=="common_14") %>%
    mutate(top_beats_random_target_logit=as.numeric(tolower(as.character(top_beats_random_target_logit)) %in% c("true","1")))
images <- function(d, cols=metrics) d %>% group_by(cohort_id, case_id, class_name, model) %>%
    summarise(across(all_of(cols), finite_mean), .groups="drop")
means <- function(d, cols=metrics) d %>% group_by(model) %>% summarise(images=n(),
    across(all_of(cols), finite_mean), .groups="drop")
k <- clean(read("artifacts/kather5k_jpi_revision/statistics/image_seed_metrics.csv"))
e <- clean(read("artifacts/crc_val_external/common_seven/faithfulness/external_faithfulness_metrics.csv"))
primary <- "primary_jointly_correct_true_class"
alltrue <- "all_image_true_class_sensitivity"
ki <- images(filter(k, cohort_name=="selected_faithfulness", analysis_family==primary))
ri <- images(filter(k, cohort_name=="random_prediction_independent", analysis_family==primary))
ei <- images(filter(e, analysis_family==primary))
ea <- filter(e, analysis_family==alltrue)
eligible <- ea %>% group_by(cohort_id, seed) %>% summarise(keep=all(abs(unperturbed_target_logit)>=0.25), .groups="drop")
ef <- images(inner_join(ea, filter(eligible,keep), by=c("cohort_id","seed")))
eu <- images(ea)
# Only the ratio uses the denominator filter. All defined correlations and all
# top-vs-random indicators contribute to the unfiltered image-level endpoints.
eb <- eu %>% select(-all_of(metrics[3])) %>% left_join(
    ef %>% select(cohort_id,model,all_of(metrics[3])), by=c("cohort_id","model"))
scopes <- list(Kather_selected=ki, Kather_random=ri, CRC_conditional=ei, CRC_class_complete=eb)
summaries <- bind_rows(lapply(names(scopes), function(s) mutate(means(scopes[[s]]), scope=s)))
write(summaries, "image_weighted_model_means")
write(bind_rows(lapply(names(scopes), function(s) mutate(scopes[[s]],scope=s))), "image_level_metrics")
pairs <- list(c("UNI","ResNet18"), c("UNI","DINOv2"), c("DINOv2","ResNet18"))
set.seed(20270915)
infer <- function(d, scope, cols=metrics) {
    rows <- list()
    for(m in cols) for(pair in pairs) {
        wide <- d %>% select(cohort_id,case_id,model,all_of(m)) %>%
            pivot_wider(names_from=model,values_from=all_of(m))
        z <- wide[[pair[1]]] - wide[[pair[2]]]
        keep <- is.finite(z); z<-z[keep]; groups<-wide$case_id[keep]
        if(startsWith(scope,"Kather")) {
            bygroup <- split(z,groups); gm<-vapply(bygroup,mean,0.0)
            signs<-as.matrix(expand.grid(rep(list(c(-1,1)),length(gm))))
            null<-as.vector(signs %*% gm/length(gm)); effect<-mean(gm)
            p<-mean(abs(null)>=abs(effect)-1e-12)
            # Equal-source estimand, preserving image-level resampling within source.
            boot<-replicate(5000,mean(vapply(sample(seq_along(bygroup),length(bygroup),TRUE),
                function(g) mean(sample(bygroup[[g]],length(bygroup[[g]]),TRUE)),0.0)))
            unit<-"source_weighted_exact_sign_flip"
        } else {
            effect<-mean(z)
            p<-if(all(z==0)) 1 else suppressWarnings(wilcox.test(z,exact=FALSE,correct=FALSE)$p.value)
            boot<-replicate(5000,mean(sample(z,length(z),TRUE)))
            unit<-"tile_weighted_signed_rank_exploratory"
        }
        ci<-quantile(boot,c(.025,.975))
        rows[[length(rows)+1]]<-data.frame(scope=scope,metric=m,model_a=pair[1],model_b=pair[2],
            images=length(z),sources=length(unique(groups)),effect=effect,image_weighted_effect=mean(z),
            ci_low=ci[1],ci_high=ci[2],p=p,inference=unit)
    }
    bind_rows(rows) %>% mutate(family=ifelse(grepl("probability",metric),"probability_sensitivity",
        ifelse(metric==metrics[3],"secondary_relative_logit","primary_spearman_success"))) %>%
        group_by(scope,family) %>% mutate(p_holm=p.adjust(p,"holm")) %>% ungroup()
}
tests<-bind_rows(lapply(names(scopes),function(s) infer(scopes[[s]],s)))
write(tests,"paired_inference")
source_means<-bind_rows(lapply(c("Kather_selected","Kather_random"),function(s)
    scopes[[s]] %>% group_by(case_id,model) %>% summarise(across(all_of(metrics),finite_mean),.groups="drop") %>% mutate(scope=s)))
write(source_means,"source_group_means")
# Internal consistency: reported image-weighted effects equal differences of means.
for(i in seq_len(nrow(tests))) {
    t<-tests[i,]; a<-summaries %>% filter(scope==t$scope,model==t$model_a)
    b<-summaries %>% filter(scope==t$scope,model==t$model_b)
    stopifnot(abs(a[[t$metric]]-b[[t$metric]]-t$image_weighted_effect)<1e-10)
}
counts<-ea %>% group_by(model) %>% summarise(rows=n(),defined=sum(is.finite(attribution_occlusion_spearman)),images=n_distinct(cohort_id),.groups="drop")
write(counts,"external_unfiltered_spearman_counts")

# Audit the seven-class taxonomy and saved predictions, independently of model code.
pred<-read("artifacts/crc_val_external/common_seven/classification/all_external_predictions_and_logits.csv")
classes<-c("adipose","background","debris_mucus","lymphocytes","normal_mucosa","stroma","tumor")
mapping<-c(ADI="adipose",BACK="background",DEB="debris_mucus",MUC="debris_mucus",LYM="lymphocytes",NORM="normal_mucosa",STR="stroma",TUM="tumor")
logit_cols<-paste0("logit_",0:6,"_",classes)
argmax<-max.col(as.matrix(pred[,logit_cols]),ties.method="first")-1
stopifnot(all(pred$class_name==unname(mapping[pred$source_class])),
          all(pred$label==match(pred$class_name,classes)-1),all(pred$prediction==argmax),
          all(pred$predicted_class_name==classes[pred$prediction+1]),
          !any(duplicated(pred[c("model","seed","relative_path")])) )
checks<-pred %>% group_by(model,seed) %>% summarise(tiles=n(),class_count=n_distinct(label),.groups="drop")
stopifnot(all(checks$tiles==6588),all(checks$class_count==7))
write(checks,"external_mapping_audit")
subclass<-pred %>% group_by(model,seed,source_class) %>%
    summarise(tiles=n(),recall=mean(prediction==label),.groups="drop")
write(subclass,"external_source_class_per_seed")
submeans<-subclass %>% group_by(model,source_class) %>% summarise(tiles=first(tiles),recall=mean(recall),.groups="drop")
write(submeans,"external_source_class_means")
errors<-pred %>% filter(source_class=="LYM",seed==11) %>% group_by(model) %>%
    mutate(selection_priority=ifelse(model=="DINOv2",prediction==label,prediction!=label)) %>%
    arrange(desc(selection_priority),desc(confidence),relative_path,.by_group=TRUE) %>% slice_head(n=4) %>% ungroup()
write(errors,"lymphocyte_example_manifest")

theme_set(theme_minimal(base_size=11)+theme(panel.grid.minor=element_blank(),legend.position="bottom"))
colors<-c(ResNet18="#3266A8",DINOv2="#D17A22",UNI="#23866B")
plotdata<-summaries %>% filter(scope!="Kather_random") %>% pivot_longer(all_of(metrics),names_to="metric",values_to="value") %>%
    mutate(endpoint=factor(short[metric],levels=short),scope=factor(scope,levels=c("Kather_selected","CRC_conditional","CRC_class_complete"),
    labels=c("Kather conditional","CRC conditional","CRC class-complete")))
g<-ggplot(plotdata,aes(scope,value,color=model,group=model))+geom_hline(yintercept=0,color="grey70")+
    geom_line()+geom_point(size=2.5)+facet_wrap(~endpoint,scales="free_y",nrow=1)+scale_color_manual(values=colors)+
    labs(x=NULL,y=NULL,color=NULL,title="Faithfulness across datasets and evaluation protocols",
    subtitle="Equal image weighting after eligible-seed averaging; relative-logit ratios are secondary")+
    theme(axis.text.x=element_text(angle=20,hjust=1))
ggsave(file.path(fig,"figure_feedback2_cross_dataset.pdf"),g,width=11,height=4)
ggsave(file.path(fig,"figure_feedback2_cross_dataset.png"),g,width=11,height=4,dpi=300)

kg<-tests %>% filter(scope=="Kather_selected",metric!=metrics[3]) %>%
    mutate(endpoint=short[metric],contrast=paste(model_a,"minus",model_b))
kp<-ggplot(kg,aes(effect,contrast))+geom_vline(xintercept=0,color="grey60",linetype=2)+
    geom_segment(aes(x=ci_low,xend=ci_high,yend=contrast),linewidth=.7)+geom_point(size=2.6,color="#23866B")+
    facet_wrap(~endpoint,scales="free_x")+labs(x="Source-weighted paired effect (95% hierarchical CI)",y=NULL,
    title="Principal Kather evidence: equal source-group weighting")
ggsave(file.path(fig,"figure_feedback2_kather.pdf"),kp,width=10,height=3.5)
byclass<-eb %>% group_by(class_name,model) %>% summarise(across(all_of(metrics),finite_mean),.groups="drop")
write(byclass,"external_class_heterogeneity")
hg<-byclass %>% pivot_longer(all_of(metrics[1:2]),names_to="metric",values_to="value")
hp<-ggplot(hg,aes(class_name,value,color=model,group=model))+geom_hline(yintercept=0,color="grey60")+
    geom_point()+geom_line()+facet_wrap(~metric,scales="free_y",labeller=as_labeller(short))+
    scale_color_manual(values=colors)+labs(x=NULL,y=NULL,color=NULL,title="External class-complete sensitivity: no denominator filter")+
    theme(axis.text.x=element_text(angle=40,hjust=1))
ggsave(file.path(fig,"figure_feedback2_heterogeneity.pdf"),hp,width=10,height=4)

# Raw-tile gallery; no pathological relabeling or image manipulation.
chosen<-unique(unlist(lapply(c("ResNet18","UNI","DINOv2"),function(m)
    head(errors$relative_path[errors$model==m],2))))
draw_gallery<-function() {
    grid::grid.newpage()
    grid::grid.text("External lymphocyte audit | original tiles, seed 11",x=.5,y=.975,
                    gp=grid::gpar(fontsize=16,fontface="bold"))
    for(i in seq_along(chosen)) {
        x<-((i-1)%%3+.5)/3; y<-.74-floor((i-1)/3)*.46
        im<-magick::image_read(file.path(root,"CRC-VAL-HE-7K",chosen[i]))
        grid::grid.raster(as.raster(im),x=x,y=y,width=grid::unit(2.3,"in"),height=grid::unit(2.3,"in"),interpolate=FALSE)
        grid::grid.text(sub("LYM-TCGA-","LYM ",basename(chosen[i])),x=x,y=y+.17,gp=grid::gpar(fontsize=11))
        labels<-vapply(models,function(m) {
            row<-filter(pred,relative_path==chosen[i],seed==11,model==m)
            sprintf("%s: %s (%.2f)",m,gsub("_"," ",row$predicted_class_name),row$confidence)
        },"")
        grid::grid.text(paste(labels,collapse="\n"),x=x,y=y-.21,gp=grid::gpar(fontsize=10))
    }
}
if (dir.exists(file.path(root,"CRC-VAL-HE-7K"))) {
pdf(file.path(fig,"figure_feedback2_lymphocyte_audit.pdf"),width=10,height=8); draw_gallery(); dev.off()
png(file.path(fig,"figure_feedback2_lymphocyte_audit.png"),width=2000,height=1600,res=200); draw_gallery(); dev.off()
} else message("Skipping tile gallery: original CRC images not installed.")

# Compact tables included by LaTeX; numeric values are generated, not transcribed.
tex_escape<-function(x) gsub("_","\\_",x,fixed=TRUE)
table_tex<-function(d,caption,label,file) {
    lines<-c("\\begin{table}[htbp]","\\centering\\small",paste0("\\caption{",caption,"}\\label{",label,"}"),
        paste0("\\begin{tabular}{",paste(rep("l",ncol(d)),collapse=""),"}\\toprule"),
        paste0(paste(names(d),collapse=" & ")," \\\\"),"\\midrule",
        apply(d,1,function(r) paste0(paste(r,collapse=" & ")," \\\\")),"\\bottomrule\\end{tabular}","\\end{table}")
    writeLines(lines,file.path(paper,file))
}
f<-function(x) sprintf("%.3f",x)
tab<-summaries %>% filter(scope=="CRC_conditional") %>%
    transmute(Model=model,Spearman=f(.data[[metrics[1]]]),`Relative AUC`=f(.data[[metrics[3]]]),`Top beats random`=f(.data[[metrics[2]]]))
table_tex(tab,"Conditional external faithfulness: 48 tiles and 106 eligible image--seed observations per model. Each endpoint is averaged over eligible seeds within tile and then equally across tiles. Relative-logit AUC is secondary and is not shift-invariant.","tab:external_faithfulness","feedback2_external_table.tex")
randomtab<-summaries %>% filter(scope=="Kather_random") %>% transmute(Model=model,Images=images,Spearman=f(.data[[metrics[1]]]),`Top beats random`=f(.data[[metrics[2]]]),`Relative AUC`=f(.data[[metrics[3]]]))
table_tex(randomtab,"Prediction-independent 128-image Kather cohort, conditional jointly-correct subset. The Images column gives the number with at least one eligible seed. Means weight images equally. Relative AUC is secondary.","tab:random_means","feedback2_random_means.tex")
rt<-filter(tests,scope=="Kather_random",metric!=metrics[3]) %>% transmute(Endpoint=short[metric],Contrast=paste(model_a,"--",model_b),Effect=f(effect),`95\\% CI`=sprintf("[%.3f, %.3f]",ci_low,ci_high),`Holm $p$`=sprintf("%.4f",p_holm))
table_tex(rt,"Random-cohort source-weighted contrasts and hierarchical source/image bootstrap intervals. Holm correction covers six tests: three model pairs for Spearman and deletion success. Source effects are equally weighted.","tab:random_tests","feedback2_random_tests.tex")
pt<-filter(tests,scope=="Kather_selected",metric!=metrics[3]) %>% transmute(Endpoint=short[metric],Contrast=paste(model_a,"--",model_b),Effect=f(effect),`95\\% CI`=sprintf("[%.3f, %.3f]",ci_low,ci_high),`Holm $p$`=sprintf("%.4f",p_holm))
table_tex(pt,"Principal source-weighted inference for the selected Kather cohort. Exact sign-flip tests enumerate all 1,024 source-sign configurations. Hierarchical intervals use the same equal-source estimand. Holm correction covers the six Spearman/deletion-success contrasts.","tab:source_primary","feedback2_source_tests.tex")
st<-submeans %>% filter(source_class %in% c("LYM","DEB","MUC")) %>% mutate(recall=f(recall)) %>% pivot_wider(names_from=source_class,values_from=c(recall,tiles)) %>% select(model,recall_LYM,recall_DEB,recall_MUC)
names(st)<-c("Model","LYM recall","DEB recall","MUC recall")
table_tex(st,"External recall by original source category, averaged over three seeds. LYM has 634 tiles, DEB 339, and MUC 1,035; DEB and MUC share the debris/mucus prediction target, so these are not separate nine-class predictions.","tab:crc_subclasses","feedback2_subclass_table.tex")

probdir<-file.path(root,"artifacts/feedback2_revision/probability_deletion/deterministic_ties_v5/full")
probability_means<-list(); probability_tests<-list()
for(dataset in c("kather","crc")) {
    pfile<-file.path(probdir,dataset,"probability_image_seed_metrics.csv")
    if(file.exists(pfile)) {
        p<-read.csv(pfile)
        stopifnot(all(p$probability_protocol=="deterministic_ties_v5_sensitivity"),
                  all(p$tie_policy=="score_then_patch_index_ascending"))
        cols<-c("top_minus_random_probability_reduction_auc","top_beats_random_probability")
        eligible<-if(dataset=="kather") filter(k,analysis_family==primary) else filter(e,analysis_family %in% c(primary,alltrue))
        id<-c("cohort_name","cohort_id","model","seed","fold","target_class","analysis_family")
        stopifnot(nrow(anti_join(eligible,p,by=id))==0,!any(duplicated(p[id])))
        for(cohort in unique(p$cohort_name)) for(family in intersect(unique(p$analysis_family),c(primary,alltrue))) {
            block<-filter(p,cohort_name==cohort,analysis_family==family)
            if(nrow(block)==0) next
            pi<-images(block,cols)
            scope<-paste(if(dataset=="kather") "Kather" else "CRC",cohort,family,sep="_")
            pm<-mutate(means(pi,cols),scope=scope)
            pt<-infer(pi,scope,cols)
            write(pm,paste0(scope,"_probability_means"))
            write(pt,paste0(scope,"_probability_tests"))
            probability_means[[scope]]<-pm; probability_tests[[scope]]<-pt
        }
    }
}
if(length(probability_means)) {
    write(bind_rows(probability_means),"probability_model_means")
    write(bind_rows(probability_tests),"probability_paired_inference")
}
missing_probability<-c("kather","crc")[!file.exists(file.path(probdir,c("kather","crc"),"probability_image_seed_metrics.csv"))]
probability_status<-if(length(missing_probability)) paste("Probability sensitivity pending for:",paste(missing_probability,collapse=", ")) else "Full probability sensitivity tables generated; manuscript interpretation requires review."
writeLines(c("CPU reporting complete.",probability_status,
             "No model retraining or attribution regeneration was performed."),file.path(out,"status.txt"))
print(summaries); print(tests %>% select(scope,metric,model_a,model_b,effect,ci_low,ci_high,p_holm))
