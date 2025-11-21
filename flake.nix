{
  description = "Water quality python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    micromamba-shell.url = "github:vikineema/micromamba-shell";
  };

  outputs =
    {
      self,
      nixpkgs,
      micromamba-shell,
      ...
    }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          micromamba-shell.packages.${system}.default
          pkgs.actionlint
          pkgs.jq
          pkgs.moreutils
          pkgs.markdownlint-cli
          pkgs.nixfmt-rfc-style
          pkgs.nodePackages.cspell
        ];
        shellHook = ''
          export TMPDIR=$HOME/.tmp
          export PROD_YAML=${./docker/environment_prod.yaml}
          export PROD_REQS=${./docker/requirements_prod.txt}
          export DEV_YAML=${./docker/environment_dev.yaml}
          export DEV_REQS=${./docker/requirements_dev.txt}

          if [ -z "$MICROMAMBA_EXE" ]; then
            echo "Entering default micromamba shell..."
            exec micromamba-shell --rcfile ${./micromamba_shell_hook.sh}
          fi
        '';
      };
    };
}
