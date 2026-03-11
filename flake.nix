{
  description = "Water quality python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11"; # Updated to match your other config
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
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      pkgsFor = forAllSystems (system: import nixpkgs {
        localSystem = system;
        config.allowUnfree = false;
      });
    in
    {
      
      devShells = forAllSystems (system: 
        let
          pkgs = pkgsFor.${system};
        in 
        {
          default = pkgs.mkShell {
            buildInputs = [
              # Accesses the micromamba-shell package for the current system
              micromamba-shell.packages.${system}.default
              pkgs.actionlint
              pkgs.jq
              pkgs.moreutils
              pkgs.markdownlint-cli
              pkgs.nixfmt-rfc-style
              pkgs.nodePackages.cspell
              pkgs.yq
            ];

            shellHook = ''
              export TMPDIR=$HOME/.tmp
              export ENV_YAML=${./environment.yaml}
              export REQS_TXT=${./requirements.txt}
              
              if [ -z "$MICROMAMBA_EXE" ]; then
                echo "Entering default micromamba shell..."
                exec micromamba-shell --rcfile ${./micromamba_shell_hook.sh}
              fi
            '';
          };
        }
      );
    };
}