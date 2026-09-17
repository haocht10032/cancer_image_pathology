suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr)})
args<-commandArgs(trailingOnly=TRUE)
root<-if(length(args)) args[1] else getwd()
out<-file.path(root,"artifacts/feedback2_revision/reporting")
d<-read.csv(file.path(out,"probability_model_means.csv"))
d<-d %>% mutate(cohort=case_when(
    grepl("Kather_selected",scope)~"Kather: selected, n = 187",
    grepl("Kather_random",scope)~"Kather: random, n = 100",
    grepl("all_image",scope)~"CRC: class-complete, n = 98",
    TRUE~"CRC: jointly correct, n = 48"),
    cohort=factor(cohort,levels=c("Kather: selected, n = 187","Kather: random, n = 100",
                                 "CRC: jointly correct, n = 48","CRC: class-complete, n = 98")),
    model=factor(model,levels=c("ResNet18","DINOv2","UNI"))) %>%
    pivot_longer(c(top_minus_random_probability_reduction_auc,top_beats_random_probability),
                 names_to="endpoint",values_to="value") %>%
    mutate(endpoint=factor(endpoint,
        levels=c("top_minus_random_probability_reduction_auc","top_beats_random_probability"),
        labels=c("Top-minus-random\nprobability-drop AUC","Top deletion beats random:\nproportion")),
        cohort=factor(cohort,levels=levels(cohort),labels=c("Kather selected\nn = 187",
            "Kather random\nn = 100","CRC jointly correct\nn = 48","CRC class-complete\nn = 98")))
g<-ggplot(d,aes(value,model,color=model))+geom_vline(xintercept=0,color="grey75",linewidth=.4)+
    geom_point(size=3)+facet_grid(cohort~endpoint,scales="free_x")+
    scale_color_manual(values=c(ResNet18="#3266A8",DINOv2="#D17A22",UNI="#23866B"))+
    labs(x=NULL,y=NULL,title="Probability-deletion sensitivity",
         subtitle="Image-weighted means after eligible-seed averaging;\ndeterministic patch-index ties",
         caption="Larger values favor the attribution ranking. AUC integrates fractions 0-0.5.\nDescriptive means; paired inference and intervals are reported separately.")+
    theme_minimal(base_size=12)+theme(legend.position="none",panel.grid.minor=element_blank(),
        panel.grid.major.y=element_blank(),strip.text.y=element_text(angle=0),
        plot.caption=element_text(hjust=0),plot.title=element_text(face="bold"))
ggsave(file.path(out,"probability_summary.png"),g,width=8,height=8,dpi=200,bg="white")
ggsave(file.path(root,"analysis/manuscript_figures/figure_feedback2_probability.pdf"),
       g,width=8,height=8,bg="white")
