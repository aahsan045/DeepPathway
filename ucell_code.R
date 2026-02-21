library(Matrix)
library(R.matlab)
library(UCell)
library(jsonlite)
json_data <- fromJSON("/home/e90244aa/Bleep/DeepPathwayV2/prostate cancer dataset/pathway_dict_20.json")
signatures_list <- lapply(json_data, as.character)
signatures_list <- as.list(signatures_list)
names <- c("MEND61","MEND62")
for (name in names){
  link <- paste0("/home/e90244aa/Bleep/DeepPathwayV2/prostate cancer dataset/data for Ucell calculations/",name,"_data.mat")
  sample <- readMat(link)
  data <- sample$x
  cell_names <- unlist(sample$cell.names)
  gene_names <- unlist(sample$gene.names)
  sparse_data <- as(data, "CsparseMatrix")
  DgCMatrix_obj <- as(sparse_data, "dgCMatrix")
  colnames(DgCMatrix_obj) <- cell_names
  gene_names <- as.vector(trimws(unlist(sample$gene.names)))
  rownames(DgCMatrix_obj) <- gene_names
  scores <- ScoreSignatures_UCell(DgCMatrix_obj, features=signatures_list,maxRank=1543)
  link <- paste0("/home/e90244aa/Bleep/DeepPathwayV2/prostate cancer dataset/pathway expression/",name,"_pathway expression.csv")
  write.csv(scores, file = link, row.names = TRUE)
}
