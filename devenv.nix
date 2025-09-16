{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = with pkgs; [ 
    git entr findutils 
    ];

  # https://devenv.sh/languages/
  languages.texlive = {
    enable = true;
    packages = with pkgs.texlive;
      [ 
        "babel-spanish" "physics" "amsfonts"
        "siunits" 
        
        ### Estos son los paquetes necesarios para
        ### compilar el formato de Elsevier
        "elsarticle" "lipsum"
       ];
  };
  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.watch-latex.exec = ''
    find . -name "*.tex" -or -name "*.bib" \
    | entr -s "pdflatex -interaction=nonstopmode grb.tex && \
     bibtex grb && \
     pdflatex -interaction=nonstopmode grb.tex && \
     pdflatex -interaction=nonstopmode grb.tex && \
     echo 'Compilation completed.'"
  '';

  enterShell = ''
    hello
    git --version
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
